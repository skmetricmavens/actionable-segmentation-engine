"""
Tests for CLV feature engineering module.
"""

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from src.data.schemas import (
    BehaviorType,
    CategoryAffinity,
    CustomerProfile,
    PurchaseIntervalMetrics,
)
from src.features.clv.features import (
    CLVFeatureBuilder,
    CLVFeatureConfig,
    extract_clv_features,
    extract_clv_feature_matrix,
)


# =============================================================================
# FIXTURES
# =============================================================================


def make_profile(
    customer_id: str = "test_customer",
    *,
    total_purchases: int = 5,
    total_revenue: Decimal = Decimal("500"),
    avg_order_value: Decimal = Decimal("100"),
    days_since_last_purchase: int = 15,
    purchase_frequency_per_month: float = 1.5,
    total_sessions: int = 20,
    total_page_views: int = 100,
    total_items_viewed: int = 50,
    total_cart_additions: int = 10,
    cart_abandonment_rate: float = 0.5,
    total_refunds: int = 1,
    refund_rate: float = 0.2,
    churn_risk_score: float = 0.3,
    preferred_day_of_week: int | None = 2,  # Wednesday
    preferred_hour_of_day: int | None = 14,  # 2 PM
    mobile_session_ratio: float = 0.4,
    behavior_type: BehaviorType = BehaviorType.REGULAR,
    category_affinities: list[CategoryAffinity] | None = None,
    purchase_intervals: PurchaseIntervalMetrics | None = None,
    first_seen: datetime | None = None,
    last_seen: datetime | None = None,
) -> CustomerProfile:
    """Create a CustomerProfile for testing."""
    if first_seen is None:
        first_seen = datetime(2024, 1, 1, tzinfo=timezone.utc)
    if last_seen is None:
        last_seen = datetime(2024, 6, 15, tzinfo=timezone.utc)  # Mid-year
    if category_affinities is None:
        category_affinities = [
            CategoryAffinity(category="Electronics", engagement_score=0.8, view_count=30, purchase_count=3, level=1),
            CategoryAffinity(category="Books", engagement_score=0.4, view_count=15, purchase_count=1, level=1),
        ]

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
        total_refunds=total_refunds,
        refund_rate=refund_rate,
        churn_risk_score=churn_risk_score,
        preferred_day_of_week=preferred_day_of_week,
        preferred_hour_of_day=preferred_hour_of_day,
        mobile_session_ratio=mobile_session_ratio,
        behavior_type=behavior_type,
        category_affinities=category_affinities,
        purchase_intervals=purchase_intervals,
    )


# =============================================================================
# FEATURE BUILDER TESTS
# =============================================================================


class TestCLVFeatureBuilder:
    """Tests for CLVFeatureBuilder class."""

    def test_default_config(self) -> None:
        """Test builder with default configuration."""
        builder = CLVFeatureBuilder()
        profile = make_profile()

        features = builder.build_features(profile)

        # Should have features from all categories
        assert len(features) > 30
        assert "total_revenue" in features
        assert "total_sessions" in features
        assert "interval_mean" in features

    def test_feature_names_consistent(self) -> None:
        """Test that feature names are consistent across calls."""
        builder = CLVFeatureBuilder()
        profile1 = make_profile(customer_id="cust1")
        profile2 = make_profile(customer_id="cust2", total_revenue=Decimal("1000"))

        features1 = builder.build_features(profile1)
        features2 = builder.build_features(profile2)

        # Feature names should be identical
        assert list(features1.keys()) == list(features2.keys())
        # But values should differ
        assert features1["total_revenue"] != features2["total_revenue"]

    def test_n_features_property(self) -> None:
        """Test n_features property."""
        builder = CLVFeatureBuilder()
        profile = make_profile()
        builder.build_features(profile)

        assert builder.n_features > 0
        assert builder.n_features == len(builder.feature_names)


class TestRFMFeatures:
    """Tests for RFM feature extraction."""

    def test_rfm_features_extracted(self) -> None:
        """Test that RFM features are correctly extracted."""
        profile = make_profile(
            total_purchases=10,
            total_revenue=Decimal("1000"),
            avg_order_value=Decimal("100"),
            days_since_last_purchase=7,
            purchase_frequency_per_month=2.0,
        )
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["total_purchases"] == 10.0
        assert features["total_revenue"] == 1000.0
        assert features["avg_order_value"] == 100.0
        assert features["days_since_last_purchase"] == 7.0
        assert features["purchase_frequency_per_month"] == 2.0

    def test_missing_days_since_purchase(self) -> None:
        """Test handling of None days_since_last_purchase."""
        profile = make_profile(days_since_last_purchase=None)
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        # Should use sentinel value
        assert features["days_since_last_purchase"] == -1.0

    def test_refund_metrics(self) -> None:
        """Test refund-related features."""
        profile = make_profile(total_refunds=3, refund_rate=0.6)
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["total_refunds"] == 3.0
        assert features["refund_rate"] == 0.6


class TestEngagementFeatures:
    """Tests for engagement feature extraction."""

    def test_engagement_features_extracted(self) -> None:
        """Test that engagement features are correctly extracted."""
        profile = make_profile(
            total_sessions=25,
            total_page_views=150,
            total_items_viewed=75,
            total_cart_additions=15,
            cart_abandonment_rate=0.4,
        )
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["total_sessions"] == 25.0
        assert features["total_page_views"] == 150.0
        assert features["total_items_viewed"] == 75.0
        assert features["total_cart_additions"] == 15.0
        assert features["cart_abandonment_rate"] == 0.4


class TestIntervalFeatures:
    """Tests for purchase interval feature extraction."""

    def test_interval_features_with_data(self) -> None:
        """Test interval features when data is available."""
        intervals = PurchaseIntervalMetrics(
            interval_mean=30.0,
            interval_std=5.0,
            interval_min=25.0,
            interval_max=35.0,
            interval_cv=0.17,
            regularity_index=0.92,
        )
        profile = make_profile(purchase_intervals=intervals)
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["interval_mean"] == 30.0
        assert features["interval_std"] == 5.0
        assert features["interval_cv"] == 0.17
        assert features["regularity_index"] == 0.92

    def test_interval_features_without_data(self) -> None:
        """Test interval features when no interval data available."""
        profile = make_profile(purchase_intervals=None)
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        # Should use sentinel values
        assert features["interval_mean"] == -1.0
        assert features["interval_std"] == -1.0
        assert features["interval_cv"] == -1.0
        assert features["regularity_index"] == -1.0


class TestTemporalFeatures:
    """Tests for temporal feature extraction."""

    def test_tenure_days(self) -> None:
        """Test customer tenure calculation."""
        profile = make_profile(
            first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
            last_seen=datetime(2024, 4, 1, tzinfo=timezone.utc),  # 91 days
        )
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["tenure_days"] == 91.0

    def test_preferred_time_features(self) -> None:
        """Test preferred day/hour features."""
        profile = make_profile(
            preferred_day_of_week=5,  # Saturday
            preferred_hour_of_day=10,
        )
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["preferred_dow"] == 5.0
        assert features["preferred_hour"] == 10.0
        assert features["prefers_weekend"] == 1.0  # Saturday is weekend

    def test_weekday_preference(self) -> None:
        """Test weekday preference indicator."""
        profile = make_profile(preferred_day_of_week=2)  # Wednesday
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["prefers_weekend"] == 0.0


class TestEfficiencyFeatures:
    """Tests for efficiency ratio features."""

    def test_revenue_per_session(self) -> None:
        """Test revenue per session calculation."""
        profile = make_profile(
            total_revenue=Decimal("1000"),
            total_sessions=20,
        )
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["revenue_per_session"] == 50.0  # 1000 / 20

    def test_cart_to_purchase_rate(self) -> None:
        """Test cart to purchase conversion rate."""
        profile = make_profile(
            total_purchases=4,
            total_cart_additions=8,
        )
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["cart_to_purchase_rate"] == 0.5  # 4 / 8

    def test_zero_sessions_handling(self) -> None:
        """Test handling of zero sessions (avoid division by zero)."""
        profile = make_profile(total_sessions=0)
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["revenue_per_session"] == 0.0
        assert features["pages_per_session"] == 0.0
        assert features["items_per_session"] == 0.0

    def test_activity_density(self) -> None:
        """Test activity density (sessions per tenure day)."""
        profile = make_profile(
            total_sessions=30,
            first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
            last_seen=datetime(2024, 2, 1, tzinfo=timezone.utc),  # 31 days
        )
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert abs(features["activity_density"] - 30/31) < 0.01


class TestCategoryFeatures:
    """Tests for category affinity features."""

    def test_category_features_with_affinities(self) -> None:
        """Test category features when affinities exist."""
        affinities = [
            CategoryAffinity(category="Electronics", engagement_score=0.8, view_count=40, purchase_count=4, level=1),
            CategoryAffinity(category="Books", engagement_score=0.2, view_count=10, purchase_count=1, level=1),
        ]
        profile = make_profile(category_affinities=affinities)
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["category_count"] == 2.0
        assert features["top_category_score"] == 0.8
        assert features["top_category_purchases"] == 4.0
        assert features["category_concentration"] == 0.8  # 0.8 / (0.8 + 0.2)

    def test_category_features_empty_affinities(self) -> None:
        """Test category features with no affinities."""
        profile = make_profile(category_affinities=[])
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["category_count"] == 0.0
        assert features["top_category_score"] == 0.0


class TestBehaviorFeatures:
    """Tests for behavior classification features."""

    def test_regular_behavior_encoding(self) -> None:
        """Test one-hot encoding for REGULAR behavior."""
        profile = make_profile(behavior_type=BehaviorType.REGULAR)
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["is_regular"] == 1.0
        assert features["is_irregular"] == 0.0
        assert features["is_long_cycle"] == 0.0
        assert features["is_one_time"] == 0.0
        assert features["is_new"] == 0.0

    def test_irregular_behavior_encoding(self) -> None:
        """Test one-hot encoding for IRREGULAR behavior."""
        profile = make_profile(behavior_type=BehaviorType.IRREGULAR)
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        assert features["is_regular"] == 0.0
        assert features["is_irregular"] == 1.0

    def test_mobile_primary_flag(self) -> None:
        """Test mobile primary indicator."""
        mobile_profile = make_profile(mobile_session_ratio=0.7)
        desktop_profile = make_profile(mobile_session_ratio=0.3)
        builder = CLVFeatureBuilder()

        mobile_features = builder.build_features(mobile_profile)
        desktop_features = builder.build_features(desktop_profile)

        assert mobile_features["is_mobile_primary"] == 1.0
        assert desktop_features["is_mobile_primary"] == 0.0


class TestSeasonalityFeatures:
    """Tests for seasonality feature extraction."""

    def test_month_encoding(self) -> None:
        """Test month sin/cos encoding."""
        # June (month 6)
        profile = make_profile(last_seen=datetime(2024, 6, 15, tzinfo=timezone.utc))
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        # Sin/cos of 6/12 * 2*pi
        expected_sin = np.sin(2 * np.pi * 6 / 12)
        expected_cos = np.cos(2 * np.pi * 6 / 12)

        assert abs(features["last_month_sin"] - expected_sin) < 0.001
        assert abs(features["last_month_cos"] - expected_cos) < 0.001

    def test_dow_encoding(self) -> None:
        """Test day-of-week sin/cos encoding."""
        profile = make_profile(preferred_day_of_week=3)  # Thursday
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        expected_sin = np.sin(2 * np.pi * 3 / 7)
        expected_cos = np.cos(2 * np.pi * 3 / 7)

        assert abs(features["pref_dow_sin"] - expected_sin) < 0.001
        assert abs(features["pref_dow_cos"] - expected_cos) < 0.001

    def test_hour_encoding(self) -> None:
        """Test hour sin/cos encoding."""
        profile = make_profile(preferred_hour_of_day=18)  # 6 PM
        builder = CLVFeatureBuilder()
        features = builder.build_features(profile)

        expected_sin = np.sin(2 * np.pi * 18 / 24)
        expected_cos = np.cos(2 * np.pi * 18 / 24)

        assert abs(features["pref_hour_sin"] - expected_sin) < 0.001
        assert abs(features["pref_hour_cos"] - expected_cos) < 0.001


class TestFeatureMatrix:
    """Tests for batch feature extraction."""

    def test_build_feature_matrix(self) -> None:
        """Test building feature matrix from multiple profiles."""
        profiles = [
            make_profile(customer_id="cust1", total_revenue=Decimal("100")),
            make_profile(customer_id="cust2", total_revenue=Decimal("200")),
            make_profile(customer_id="cust3", total_revenue=Decimal("300")),
        ]
        builder = CLVFeatureBuilder()

        matrix, feature_names = builder.build_feature_matrix(profiles)

        assert matrix.shape[0] == 3  # 3 profiles
        assert matrix.shape[1] == len(feature_names)
        assert "total_revenue" in feature_names

        # Check values are in correct order
        revenue_idx = feature_names.index("total_revenue")
        assert matrix[0, revenue_idx] == 100.0
        assert matrix[1, revenue_idx] == 200.0
        assert matrix[2, revenue_idx] == 300.0

    def test_empty_profiles_list(self) -> None:
        """Test handling empty profiles list."""
        builder = CLVFeatureBuilder()
        matrix, feature_names = builder.build_feature_matrix([])

        assert matrix.shape == (0, 0)
        assert feature_names == []


class TestFeatureConfig:
    """Tests for feature configuration."""

    def test_disable_category_features(self) -> None:
        """Test disabling specific feature categories."""
        config = CLVFeatureConfig(include_category=False)
        builder = CLVFeatureBuilder(config)
        profile = make_profile()

        features = builder.build_features(profile)

        assert "category_count" not in features
        assert "top_category_score" not in features
        assert "total_revenue" in features  # RFM still included

    def test_disable_seasonality(self) -> None:
        """Test disabling seasonality features."""
        config = CLVFeatureConfig(include_seasonality=False)
        builder = CLVFeatureBuilder(config)
        profile = make_profile()

        features = builder.build_features(profile)

        assert "last_month_sin" not in features
        assert "pref_dow_sin" not in features

    def test_minimal_config(self) -> None:
        """Test with only RFM features enabled."""
        config = CLVFeatureConfig(
            include_rfm=True,
            include_engagement=False,
            include_intervals=False,
            include_temporal=False,
            include_efficiency=False,
            include_category=False,
            include_behavior=False,
            include_seasonality=False,
        )
        builder = CLVFeatureBuilder(config)
        profile = make_profile()

        features = builder.build_features(profile)

        # Should only have RFM features
        assert "total_revenue" in features
        assert "total_purchases" in features
        assert "total_sessions" not in features  # Engagement disabled
        assert "interval_mean" not in features  # Intervals disabled


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_extract_clv_features(self) -> None:
        """Test convenience function for single profile."""
        profile = make_profile()
        features = extract_clv_features(profile)

        assert isinstance(features, dict)
        assert "total_revenue" in features

    def test_extract_clv_feature_matrix(self) -> None:
        """Test convenience function for multiple profiles."""
        profiles = [make_profile(), make_profile()]
        matrix, names = extract_clv_feature_matrix(profiles)

        assert matrix.shape[0] == 2
        assert len(names) > 0
