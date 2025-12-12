"""
Module: features

Purpose: Feature engineering for ML-based CLV prediction.

Extracts structured features from CustomerProfile for use in ML models.
Features are organized into categories: RFM, engagement, intervals,
temporal, efficiency, category affinity, and behavior classification.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np

from src.data.schemas import BehaviorType, CustomerProfile


@dataclass
class CLVFeatureConfig:
    """Configuration for CLV feature engineering."""

    # Feature category toggles
    include_rfm: bool = True
    include_engagement: bool = True
    include_intervals: bool = True
    include_temporal: bool = True
    include_efficiency: bool = True
    include_category: bool = True
    include_behavior: bool = True
    include_seasonality: bool = True

    # Seasonality configuration
    seasonality_period_days: int = 365

    # Category features
    max_category_features: int = 3  # Top N categories to include

    # Feature metadata
    include_feature_metadata: bool = False


@dataclass
class FeatureMetadata:
    """Metadata about a feature for explainability."""

    name: str
    category: str  # rfm, engagement, intervals, temporal, etc.
    description: str
    value_range: tuple[float, float] | None = None


class CLVFeatureBuilder:
    """Extract ML features from CustomerProfile for CLV prediction.

    Builds a feature vector from customer profile data, organized into
    logical categories that map to business concepts.

    Feature Categories:
    - RFM: Recency, Frequency, Monetary metrics
    - Engagement: Session and browsing behavior
    - Intervals: Purchase timing patterns
    - Temporal: Time-based patterns (day/hour preferences)
    - Efficiency: Derived ratios (revenue per session, etc.)
    - Category: Product category affinities
    - Behavior: Customer classification (regular/irregular/etc.)
    - Seasonality: Cyclical time encodings (sin/cos)

    Example:
        >>> builder = CLVFeatureBuilder()
        >>> features = builder.build_features(profile)
        >>> print(features.keys())
        dict_keys(['days_since_last_purchase', 'purchase_frequency', ...])
    """

    def __init__(self, config: CLVFeatureConfig | None = None) -> None:
        """Initialize feature builder.

        Args:
            config: Feature engineering configuration. Uses defaults if None.
        """
        self.config = config or CLVFeatureConfig()
        self._feature_names: list[str] = []
        self._feature_metadata: dict[str, FeatureMetadata] = {}

    @property
    def feature_names(self) -> list[str]:
        """Get ordered list of feature names."""
        return self._feature_names.copy()

    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self._feature_names)

    def build_features(self, profile: CustomerProfile) -> dict[str, float]:
        """Extract all features from a single customer profile.

        Args:
            profile: CustomerProfile to extract features from

        Returns:
            Dictionary mapping feature names to values
        """
        features: dict[str, float] = {}

        if self.config.include_rfm:
            features.update(self._extract_rfm_features(profile))

        if self.config.include_engagement:
            features.update(self._extract_engagement_features(profile))

        if self.config.include_intervals:
            features.update(self._extract_interval_features(profile))

        if self.config.include_temporal:
            features.update(self._extract_temporal_features(profile))

        if self.config.include_efficiency:
            features.update(self._extract_efficiency_features(profile))

        if self.config.include_category:
            features.update(self._extract_category_features(profile))

        if self.config.include_behavior:
            features.update(self._extract_behavior_features(profile))

        if self.config.include_seasonality:
            features.update(self._extract_seasonality_features(profile))

        # Update feature names list (first call sets the order)
        if not self._feature_names:
            self._feature_names = list(features.keys())

        return features

    def build_feature_matrix(
        self,
        profiles: list[CustomerProfile],
    ) -> tuple[np.ndarray, list[str]]:
        """Build feature matrix for multiple profiles.

        Args:
            profiles: List of CustomerProfile objects

        Returns:
            Tuple of (feature_matrix, feature_names)
            - feature_matrix: numpy array of shape (n_profiles, n_features)
            - feature_names: list of feature names in column order
        """
        if not profiles:
            return np.array([]).reshape(0, 0), []

        # Extract features for all profiles
        feature_dicts = [self.build_features(p) for p in profiles]

        # Convert to matrix
        feature_names = self._feature_names
        matrix = np.array([
            [fd.get(name, 0.0) for name in feature_names]
            for fd in feature_dicts
        ])

        return matrix, feature_names

    def _extract_rfm_features(self, profile: CustomerProfile) -> dict[str, float]:
        """Extract Recency, Frequency, Monetary features."""
        features: dict[str, float] = {}

        # Recency
        features["days_since_last_purchase"] = float(
            profile.days_since_last_purchase if profile.days_since_last_purchase is not None else -1
        )

        # Frequency
        features["total_purchases"] = float(profile.total_purchases)
        features["purchase_frequency_per_month"] = profile.purchase_frequency_per_month

        # Monetary
        features["total_revenue"] = float(profile.total_revenue)
        features["avg_order_value"] = float(profile.avg_order_value)

        # Additional RFM-adjacent
        features["total_refunds"] = float(profile.total_refunds)
        features["refund_rate"] = profile.refund_rate
        features["churn_risk_score"] = profile.churn_risk_score

        # Add metadata
        if self.config.include_feature_metadata:
            self._add_metadata("days_since_last_purchase", "rfm", "Days since last purchase")
            self._add_metadata("total_purchases", "rfm", "Total number of purchases")
            self._add_metadata("purchase_frequency_per_month", "rfm", "Purchases per month")
            self._add_metadata("total_revenue", "rfm", "Total revenue from customer")
            self._add_metadata("avg_order_value", "rfm", "Average order value")

        return features

    def _extract_engagement_features(self, profile: CustomerProfile) -> dict[str, float]:
        """Extract engagement and browsing behavior features."""
        features: dict[str, float] = {}

        features["total_sessions"] = float(profile.total_sessions)
        features["total_page_views"] = float(profile.total_page_views)
        features["total_items_viewed"] = float(profile.total_items_viewed)
        features["total_cart_additions"] = float(profile.total_cart_additions)
        features["cart_abandonment_rate"] = profile.cart_abandonment_rate

        return features

    def _extract_interval_features(self, profile: CustomerProfile) -> dict[str, float]:
        """Extract purchase interval pattern features."""
        features: dict[str, float] = {}

        intervals = profile.purchase_intervals
        if intervals is not None:
            features["interval_mean"] = intervals.interval_mean if intervals.interval_mean is not None else -1.0
            features["interval_std"] = intervals.interval_std if intervals.interval_std is not None else -1.0
            features["interval_cv"] = intervals.interval_cv if intervals.interval_cv is not None else -1.0
            features["interval_min"] = intervals.interval_min if intervals.interval_min is not None else -1.0
            features["interval_max"] = intervals.interval_max if intervals.interval_max is not None else -1.0
            features["regularity_index"] = intervals.regularity_index if intervals.regularity_index is not None else -1.0
        else:
            # No interval data - use sentinel values
            features["interval_mean"] = -1.0
            features["interval_std"] = -1.0
            features["interval_cv"] = -1.0
            features["interval_min"] = -1.0
            features["interval_max"] = -1.0
            features["regularity_index"] = -1.0

        return features

    def _extract_temporal_features(self, profile: CustomerProfile) -> dict[str, float]:
        """Extract temporal pattern features."""
        features: dict[str, float] = {}

        # Customer tenure
        tenure_days = (profile.last_seen - profile.first_seen).days
        features["tenure_days"] = float(tenure_days)

        # Preferred day of week (categorical as numeric)
        features["preferred_dow"] = float(
            profile.preferred_day_of_week if profile.preferred_day_of_week is not None else -1
        )

        # Preferred hour of day
        features["preferred_hour"] = float(
            profile.preferred_hour_of_day if profile.preferred_hour_of_day is not None else -1
        )

        # Weekend preference indicator
        is_weekend_pref = profile.preferred_day_of_week in (5, 6) if profile.preferred_day_of_week is not None else False
        features["prefers_weekend"] = 1.0 if is_weekend_pref else 0.0

        return features

    def _extract_efficiency_features(self, profile: CustomerProfile) -> dict[str, float]:
        """Extract derived efficiency ratio features."""
        features: dict[str, float] = {}

        # Revenue per session
        if profile.total_sessions > 0:
            features["revenue_per_session"] = float(profile.total_revenue) / profile.total_sessions
        else:
            features["revenue_per_session"] = 0.0

        # Pages per session
        if profile.total_sessions > 0:
            features["pages_per_session"] = profile.total_page_views / profile.total_sessions
        else:
            features["pages_per_session"] = 0.0

        # Items viewed per session
        if profile.total_sessions > 0:
            features["items_per_session"] = profile.total_items_viewed / profile.total_sessions
        else:
            features["items_per_session"] = 0.0

        # Cart to purchase conversion
        if profile.total_cart_additions > 0:
            features["cart_to_purchase_rate"] = profile.total_purchases / profile.total_cart_additions
        else:
            features["cart_to_purchase_rate"] = 0.0

        # Purchase rate (purchases per session)
        if profile.total_sessions > 0:
            features["purchase_per_session"] = profile.total_purchases / profile.total_sessions
        else:
            features["purchase_per_session"] = 0.0

        # Activity density (sessions per tenure day)
        tenure_days = (profile.last_seen - profile.first_seen).days
        if tenure_days > 0:
            features["activity_density"] = profile.total_sessions / tenure_days
        else:
            features["activity_density"] = float(profile.total_sessions)

        return features

    def _extract_category_features(self, profile: CustomerProfile) -> dict[str, float]:
        """Extract category affinity features."""
        features: dict[str, float] = {}

        # Number of categories engaged with
        features["category_count"] = float(len(profile.category_affinities))

        # Top category concentration (if available)
        if profile.category_affinities:
            # Get level 1 (top-level) affinities only
            l1_affinities = [a for a in profile.category_affinities if a.level == 1]
            if l1_affinities:
                top_affinity = l1_affinities[0]
                features["top_category_score"] = top_affinity.engagement_score
                features["top_category_purchases"] = float(top_affinity.purchase_count)

                # Category concentration: top category share of total engagement
                total_engagement = sum(a.engagement_score for a in l1_affinities)
                if total_engagement > 0:
                    features["category_concentration"] = top_affinity.engagement_score / total_engagement
                else:
                    features["category_concentration"] = 0.0
            else:
                features["top_category_score"] = 0.0
                features["top_category_purchases"] = 0.0
                features["category_concentration"] = 0.0
        else:
            features["top_category_score"] = 0.0
            features["top_category_purchases"] = 0.0
            features["category_concentration"] = 0.0

        return features

    def _extract_behavior_features(self, profile: CustomerProfile) -> dict[str, float]:
        """Extract behavior classification features."""
        features: dict[str, float] = {}

        # One-hot encode behavior type
        behavior = profile.behavior_type
        features["is_regular"] = 1.0 if behavior == BehaviorType.REGULAR else 0.0
        features["is_irregular"] = 1.0 if behavior == BehaviorType.IRREGULAR else 0.0
        features["is_long_cycle"] = 1.0 if behavior == BehaviorType.LONG_CYCLE else 0.0
        features["is_one_time"] = 1.0 if behavior == BehaviorType.ONE_TIME else 0.0
        features["is_new"] = 1.0 if behavior == BehaviorType.NEW else 0.0

        # Device features
        features["mobile_session_ratio"] = profile.mobile_session_ratio
        features["is_mobile_primary"] = 1.0 if profile.mobile_session_ratio > 0.5 else 0.0

        return features

    def _extract_seasonality_features(self, profile: CustomerProfile) -> dict[str, float]:
        """Extract cyclical seasonality features using sin/cos encoding."""
        features: dict[str, float] = {}

        # Month of last activity (sin/cos encoding for cyclical nature)
        if profile.last_seen:
            month = profile.last_seen.month
            features["last_month_sin"] = np.sin(2 * np.pi * month / 12)
            features["last_month_cos"] = np.cos(2 * np.pi * month / 12)

            # Day of year (captures seasonality within year)
            day_of_year = profile.last_seen.timetuple().tm_yday
            features["last_doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
            features["last_doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)
        else:
            features["last_month_sin"] = 0.0
            features["last_month_cos"] = 0.0
            features["last_doy_sin"] = 0.0
            features["last_doy_cos"] = 0.0

        # Preferred day of week (sin/cos for cyclical)
        if profile.preferred_day_of_week is not None:
            dow = profile.preferred_day_of_week
            features["pref_dow_sin"] = np.sin(2 * np.pi * dow / 7)
            features["pref_dow_cos"] = np.cos(2 * np.pi * dow / 7)
        else:
            features["pref_dow_sin"] = 0.0
            features["pref_dow_cos"] = 0.0

        # Preferred hour of day (sin/cos for cyclical)
        if profile.preferred_hour_of_day is not None:
            hour = profile.preferred_hour_of_day
            features["pref_hour_sin"] = np.sin(2 * np.pi * hour / 24)
            features["pref_hour_cos"] = np.cos(2 * np.pi * hour / 24)
        else:
            features["pref_hour_sin"] = 0.0
            features["pref_hour_cos"] = 0.0

        return features

    def _add_metadata(
        self,
        name: str,
        category: str,
        description: str,
        value_range: tuple[float, float] | None = None,
    ) -> None:
        """Add metadata for a feature."""
        self._feature_metadata[name] = FeatureMetadata(
            name=name,
            category=category,
            description=description,
            value_range=value_range,
        )

    def get_feature_metadata(self, name: str) -> FeatureMetadata | None:
        """Get metadata for a specific feature."""
        return self._feature_metadata.get(name)

    def get_features_by_category(self, category: str) -> list[str]:
        """Get all feature names in a category."""
        return [
            name for name, meta in self._feature_metadata.items()
            if meta.category == category
        ]


def extract_clv_features(
    profile: CustomerProfile,
    config: CLVFeatureConfig | None = None,
) -> dict[str, float]:
    """Convenience function to extract CLV features from a single profile.

    Args:
        profile: CustomerProfile to extract features from
        config: Optional feature configuration

    Returns:
        Dictionary of feature name to value
    """
    builder = CLVFeatureBuilder(config)
    return builder.build_features(profile)


def extract_clv_feature_matrix(
    profiles: list[CustomerProfile],
    config: CLVFeatureConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convenience function to extract feature matrix from multiple profiles.

    Args:
        profiles: List of CustomerProfile objects
        config: Optional feature configuration

    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    builder = CLVFeatureBuilder(config)
    return builder.build_feature_matrix(profiles)
