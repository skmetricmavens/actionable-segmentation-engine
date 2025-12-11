"""
Module: clusterer

Purpose: ML-based customer clustering for segmentation.

Uses K-Prototypes for mixed data clustering (continuous + categorical features).
K-Prototypes combines K-Means (Euclidean distance for numerical) with K-Modes
(matching dissimilarity for categorical) to handle real-world customer data.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
from numpy.typing import NDArray
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data.schemas import (
    CustomerProfile,
    Segment,
    SegmentMember,
)
from src.exceptions import ClusteringError, InsufficientDataError


@dataclass
class ClusteringResult:
    """Container for clustering results."""

    labels: NDArray[np.int64]
    centroids: NDArray[np.float64]  # Numerical centroids
    categorical_centroids: list[list[str]]  # Categorical modes
    cost: float  # K-Prototypes cost (replaces inertia)
    silhouette: float | None
    n_clusters: int
    n_samples: int
    feature_names: list[str]
    categorical_feature_names: list[str]

    # Cluster-level statistics
    cluster_sizes: dict[int, int] = field(default_factory=dict)
    cluster_stats: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class MixedFeatureMatrix:
    """Container for mixed feature extraction results (numerical + categorical)."""

    numerical_matrix: NDArray[np.float64]
    categorical_matrix: NDArray[np.object_]
    numerical_feature_names: list[str]
    categorical_feature_names: list[str]
    customer_ids: list[str]
    scaler: StandardScaler | None = None

    @property
    def combined_matrix(self) -> NDArray[np.object_]:
        """Return combined matrix with numerical first, then categorical."""
        return np.column_stack([self.numerical_matrix, self.categorical_matrix])

    @property
    def categorical_indices(self) -> list[int]:
        """Return indices of categorical columns in combined matrix."""
        n_numerical = self.numerical_matrix.shape[1]
        n_categorical = self.categorical_matrix.shape[1]
        return list(range(n_numerical, n_numerical + n_categorical))

    @property
    def all_feature_names(self) -> list[str]:
        """Return all feature names."""
        return self.numerical_feature_names + self.categorical_feature_names


def extract_mixed_features(
    profiles: list[CustomerProfile],
    *,
    include_value_features: bool = True,
    include_engagement_features: bool = True,
    include_temporal_features: bool = True,
    include_channel_features: bool = True,
    include_category_features: bool = True,
) -> MixedFeatureMatrix:
    """
    Extract mixed features (numerical + categorical) from customer profiles.

    Args:
        profiles: List of customer profiles
        include_value_features: Include revenue/CLV features (numerical)
        include_engagement_features: Include session/view features (numerical)
        include_temporal_features: Include day/hour features (categorical - cyclic)
        include_channel_features: Include device features (categorical)
        include_category_features: Include product category features (categorical)

    Returns:
        MixedFeatureMatrix with both numerical and categorical features

    Raises:
        InsufficientDataError: If no profiles provided
    """
    if not profiles:
        raise InsufficientDataError(
            "No profiles provided for feature extraction",
            required=1,
            actual=0,
            data_type="profiles",
        )

    numerical_feature_names: list[str] = []
    categorical_feature_names: list[str] = []
    customer_ids: list[str] = []
    numerical_rows: list[list[float]] = []
    categorical_rows: list[list[str]] = []

    for profile in profiles:
        customer_ids.append(profile.internal_customer_id)
        num_row: list[float] = []
        cat_row: list[str] = []

        # =================================================================
        # NUMERICAL FEATURES
        # =================================================================

        # Value features (numerical)
        if include_value_features:
            if "total_revenue" not in numerical_feature_names:
                numerical_feature_names.extend([
                    "total_revenue",
                    "avg_order_value",
                    "clv_estimate",
                    "total_purchases",
                    "purchase_frequency",
                    "churn_risk",
                ])

            num_row.extend([
                float(profile.total_revenue),
                float(profile.avg_order_value),
                float(profile.clv_estimate),
                float(profile.total_purchases),
                profile.purchase_frequency_per_month,
                profile.churn_risk_score,
            ])

        # Engagement features (numerical)
        if include_engagement_features:
            if "total_sessions" not in numerical_feature_names:
                numerical_feature_names.extend([
                    "total_sessions",
                    "total_page_views",
                    "total_items_viewed",
                    "cart_abandonment_rate",
                ])

            num_row.extend([
                float(profile.total_sessions),
                float(profile.total_page_views),
                float(profile.total_items_viewed),
                profile.cart_abandonment_rate,
            ])

        # Mobile ratio (numerical - but could be treated as category too)
        if include_channel_features:
            if "mobile_ratio" not in numerical_feature_names:
                numerical_feature_names.append("mobile_ratio")
            num_row.append(profile.mobile_session_ratio)

        # =================================================================
        # CATEGORICAL FEATURES
        # =================================================================

        # Temporal features as categorical (cyclic data - not linear!)
        if include_temporal_features:
            if "preferred_day_part" not in categorical_feature_names:
                categorical_feature_names.extend([
                    "preferred_day_part",  # weekday/weekend
                    "preferred_time_slot",  # morning/afternoon/evening/night
                ])

            # Convert day to weekday/weekend (treats cyclic nature properly)
            day = profile.preferred_day_of_week
            if day is not None:
                day_part = "weekend" if day >= 5 else "weekday"
            else:
                day_part = "unknown"

            # Convert hour to time slot
            hour = profile.preferred_hour_of_day
            if hour is not None:
                if 6 <= hour < 12:
                    time_slot = "morning"
                elif 12 <= hour < 17:
                    time_slot = "afternoon"
                elif 17 <= hour < 21:
                    time_slot = "evening"
                else:
                    time_slot = "night"
            else:
                time_slot = "unknown"

            cat_row.extend([day_part, time_slot])

        # Device type as categorical
        if include_channel_features:
            if "device_preference" not in categorical_feature_names:
                categorical_feature_names.append("device_preference")

            # Categorize mobile ratio
            mobile_ratio = profile.mobile_session_ratio
            if mobile_ratio >= 0.7:
                device_pref = "mobile_first"
            elif mobile_ratio <= 0.3:
                device_pref = "desktop_first"
            else:
                device_pref = "multi_device"

            cat_row.append(device_pref)

        # Product category affinity as categorical
        if include_category_features:
            if "top_category" not in categorical_feature_names:
                categorical_feature_names.append("top_category")

            top_cat = profile.top_category if profile.top_category else "none"
            cat_row.append(top_cat)

        # Value tier as categorical (derived from CLV)
        if include_value_features:
            if "value_tier" not in categorical_feature_names:
                categorical_feature_names.append("value_tier")

            clv = float(profile.clv_estimate)
            if clv >= 50000:
                value_tier = "premium"
            elif clv >= 10000:
                value_tier = "high"
            elif clv >= 1000:
                value_tier = "mid"
            else:
                value_tier = "entry"

            cat_row.append(value_tier)

        # Engagement tier as categorical
        if include_engagement_features:
            if "engagement_tier" not in categorical_feature_names:
                categorical_feature_names.append("engagement_tier")

            sessions = profile.total_sessions
            if sessions >= 20:
                engagement_tier = "high"
            elif sessions >= 10:
                engagement_tier = "medium"
            else:
                engagement_tier = "low"

            cat_row.append(engagement_tier)

        # Churn risk tier as categorical
        if include_value_features:
            if "churn_risk_tier" not in categorical_feature_names:
                categorical_feature_names.append("churn_risk_tier")

            churn = profile.churn_risk_score
            if churn >= 0.7:
                churn_tier = "high_risk"
            elif churn >= 0.4:
                churn_tier = "medium_risk"
            else:
                churn_tier = "low_risk"

            cat_row.append(churn_tier)

        numerical_rows.append(num_row)
        categorical_rows.append(cat_row)

    numerical_matrix = np.array(numerical_rows, dtype=np.float64)
    categorical_matrix = np.array(categorical_rows, dtype=object)

    return MixedFeatureMatrix(
        numerical_matrix=numerical_matrix,
        categorical_matrix=categorical_matrix,
        numerical_feature_names=numerical_feature_names,
        categorical_feature_names=categorical_feature_names,
        customer_ids=customer_ids,
    )


def standardize_numerical_features(
    feature_matrix: MixedFeatureMatrix,
    *,
    scaler: StandardScaler | None = None,
) -> MixedFeatureMatrix:
    """
    Standardize only the numerical features using StandardScaler.

    Args:
        feature_matrix: MixedFeatureMatrix to standardize
        scaler: Optional pre-fitted scaler (for applying to new data)

    Returns:
        New MixedFeatureMatrix with standardized numerical values
    """
    if scaler is None:
        scaler = StandardScaler()
        standardized = scaler.fit_transform(feature_matrix.numerical_matrix)
    else:
        standardized = scaler.transform(feature_matrix.numerical_matrix)

    return MixedFeatureMatrix(
        numerical_matrix=standardized,
        categorical_matrix=feature_matrix.categorical_matrix,
        numerical_feature_names=feature_matrix.numerical_feature_names,
        categorical_feature_names=feature_matrix.categorical_feature_names,
        customer_ids=feature_matrix.customer_ids,
        scaler=scaler,
    )


def cluster_customers(
    feature_matrix: MixedFeatureMatrix,
    *,
    n_clusters: int = 5,
    random_seed: int = 42,
    max_iter: int = 100,
    n_init: int = 10,
    gamma: float | None = None,
) -> ClusteringResult:
    """
    Perform K-Prototypes clustering on mixed customer features.

    K-Prototypes handles both numerical and categorical features properly:
    - Numerical: Euclidean distance (like K-Means)
    - Categorical: Matching dissimilarity (like K-Modes)

    Args:
        feature_matrix: Mixed feature matrix (numerical + categorical)
        n_clusters: Number of clusters to create
        random_seed: Random seed for reproducibility
        max_iter: Maximum iterations
        n_init: Number of initializations
        gamma: Weight for categorical features (None = auto-calculate)

    Returns:
        ClusteringResult with cluster assignments and statistics

    Raises:
        ClusteringError: If clustering fails
        InsufficientDataError: If insufficient samples
    """
    n_samples = feature_matrix.numerical_matrix.shape[0]

    if n_samples < n_clusters:
        raise InsufficientDataError(
            f"Need at least {n_clusters} samples for {n_clusters} clusters",
            required=n_clusters,
            actual=n_samples,
            data_type="samples",
        )

    if n_samples < 2:
        raise InsufficientDataError(
            "Need at least 2 samples for clustering",
            required=2,
            actual=n_samples,
            data_type="samples",
        )

    try:
        # Combine numerical and categorical for K-Prototypes
        combined_matrix = feature_matrix.combined_matrix
        categorical_indices = feature_matrix.categorical_indices

        kproto = KPrototypes(
            n_clusters=n_clusters,
            init="Cao",  # Better initialization for mixed data
            random_state=random_seed,
            max_iter=max_iter,
            n_init=n_init,
            gamma=gamma,  # None = auto-weight based on data
            verbose=0,
        )

        labels = kproto.fit_predict(combined_matrix, categorical=categorical_indices)
        labels = np.array(labels, dtype=np.int64)

        # Get centroids (numerical) and modes (categorical)
        n_numerical = feature_matrix.numerical_matrix.shape[1]
        numerical_centroids = np.array(
            [c[:n_numerical] for c in kproto.cluster_centroids_],
            dtype=np.float64,
        )
        categorical_modes = [
            [str(v) for v in c[n_numerical:]]
            for c in kproto.cluster_centroids_
        ]

        # Calculate silhouette score on numerical features only
        # (mixed silhouette is complex and less interpretable)
        silhouette = None
        if n_samples >= n_clusters + 1 and n_clusters > 1:
            try:
                silhouette = float(silhouette_score(
                    feature_matrix.numerical_matrix,
                    labels,
                ))
            except Exception:
                pass  # Silhouette calculation can fail with certain distributions

        # Calculate cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(label): int(count) for label, count in zip(unique_labels, counts)}

        # Calculate cluster statistics
        cluster_stats: dict[int, dict[str, Any]] = {}
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_numerical = feature_matrix.numerical_matrix[cluster_mask]
            cluster_categorical = feature_matrix.categorical_matrix[cluster_mask]

            stats: dict[str, Any] = {}

            # Numerical stats
            for i, feature_name in enumerate(feature_matrix.numerical_feature_names):
                stats[f"{feature_name}_mean"] = float(np.mean(cluster_numerical[:, i]))
                stats[f"{feature_name}_std"] = float(np.std(cluster_numerical[:, i]))

            # Categorical stats (mode and distribution)
            for i, feature_name in enumerate(feature_matrix.categorical_feature_names):
                values, value_counts = np.unique(cluster_categorical[:, i], return_counts=True)
                mode_idx = np.argmax(value_counts)
                stats[f"{feature_name}_mode"] = str(values[mode_idx])
                stats[f"{feature_name}_distribution"] = {
                    str(v): int(c) for v, c in zip(values, value_counts)
                }

            cluster_stats[int(cluster_id)] = stats

        return ClusteringResult(
            labels=labels,
            centroids=numerical_centroids,
            categorical_centroids=categorical_modes,
            cost=float(kproto.cost_),
            silhouette=silhouette,
            n_clusters=n_clusters,
            n_samples=n_samples,
            feature_names=feature_matrix.numerical_feature_names,
            categorical_feature_names=feature_matrix.categorical_feature_names,
            cluster_sizes=cluster_sizes,
            cluster_stats=cluster_stats,
        )

    except Exception as e:
        raise ClusteringError(
            f"K-Prototypes clustering failed: {e!s}",
            n_clusters=n_clusters,
            n_samples=n_samples,
        ) from e


def find_optimal_k(
    feature_matrix: MixedFeatureMatrix,
    *,
    k_range: tuple[int, int] = (2, 10),
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Find optimal number of clusters using cost and silhouette scores.

    Args:
        feature_matrix: Mixed feature matrix
        k_range: Range of k values to test (min, max)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with analysis results
    """
    min_k, max_k = k_range
    n_samples = feature_matrix.numerical_matrix.shape[0]

    # Adjust max_k if needed
    max_k = min(max_k, n_samples - 1)
    min_k = max(2, min_k)

    if min_k > max_k:
        return {
            "optimal_k": min_k,
            "costs": {},
            "silhouettes": {},
            "reason": "insufficient_samples",
        }

    costs: dict[int, float] = {}
    silhouettes: dict[int, float] = {}

    for k in range(min_k, max_k + 1):
        try:
            result = cluster_customers(
                feature_matrix,
                n_clusters=k,
                random_seed=random_seed,
            )
            costs[k] = result.cost
            if result.silhouette is not None:
                silhouettes[k] = result.silhouette
        except (ClusteringError, InsufficientDataError):
            continue

    # Find optimal k by silhouette score (higher is better)
    optimal_k = min_k
    if silhouettes:
        optimal_k = max(silhouettes.keys(), key=lambda k: silhouettes[k])

    return {
        "optimal_k": optimal_k,
        "costs": costs,
        "silhouettes": silhouettes,
        "reason": "silhouette_score" if silhouettes else "default",
    }


def _extract_segment_traits(
    profiles: list[CustomerProfile],
) -> tuple[list[str], dict[str, Any]]:
    """
    Extract defining traits and trait summary from cluster profiles.

    Analyzes category affinities, behavioral patterns, and value metrics
    to generate human-readable traits that describe the segment.

    Args:
        profiles: List of customer profiles in the segment

    Returns:
        Tuple of (defining_traits list, trait_summary dict)
    """
    from collections import Counter

    defining_traits: list[str] = []
    trait_summary: dict[str, Any] = {}

    if not profiles:
        return defining_traits, trait_summary

    n_profiles = len(profiles)

    # 1. Category affinity analysis
    category_counts: Counter[str] = Counter()
    category_engagement: dict[str, float] = {}

    for profile in profiles:
        if profile.top_category:
            category_counts[profile.top_category] += 1

        for affinity in profile.category_affinities:
            if affinity.category not in category_engagement:
                category_engagement[affinity.category] = 0.0
            category_engagement[affinity.category] += affinity.engagement_score

    # Top categories by customer count
    top_categories = category_counts.most_common(3)
    if top_categories:
        top_cat, top_count = top_categories[0]
        pct = (top_count / n_profiles) * 100
        if pct >= 30:
            defining_traits.append(f"Strong {top_cat} affinity ({pct:.0f}%)")
        elif pct >= 15:
            defining_traits.append(f"Moderate {top_cat} interest ({pct:.0f}%)")

        trait_summary["top_categories"] = [
            {"category": cat, "customer_count": cnt, "percentage": round(cnt / n_profiles * 100, 1)}
            for cat, cnt in top_categories
        ]

    # 2. Value-based traits
    avg_clv = sum(float(p.clv_estimate) for p in profiles) / n_profiles
    avg_aov = sum(float(p.avg_order_value) for p in profiles) / n_profiles
    avg_purchases = sum(p.total_purchases for p in profiles) / n_profiles
    avg_purchase_freq = sum(p.purchase_frequency_per_month for p in profiles) / n_profiles

    # Classify value tier
    if avg_clv >= 50000:
        defining_traits.append("Premium high-value customers")
    elif avg_clv >= 10000:
        defining_traits.append("High-value customers")
    elif avg_clv >= 1000:
        defining_traits.append("Mid-value customers")
    else:
        defining_traits.append("Entry-level value customers")

    trait_summary["value_metrics"] = {
        "avg_clv": round(avg_clv, 2),
        "avg_order_value": round(avg_aov, 2),
        "avg_total_purchases": round(avg_purchases, 1),
        "avg_purchase_frequency_monthly": round(avg_purchase_freq, 2),
    }

    # 3. Engagement-based traits
    avg_sessions = sum(p.total_sessions for p in profiles) / n_profiles
    avg_page_views = sum(p.total_page_views for p in profiles) / n_profiles
    avg_cart_abandon = sum(p.cart_abandonment_rate for p in profiles) / n_profiles

    if avg_sessions >= 20:
        defining_traits.append("Highly engaged browsers")
    elif avg_sessions >= 10:
        defining_traits.append("Moderately engaged")
    else:
        defining_traits.append("Low engagement")

    if avg_cart_abandon >= 0.7:
        defining_traits.append("High cart abandonment risk")
    elif avg_cart_abandon <= 0.2:
        defining_traits.append("Strong purchase completion")

    trait_summary["engagement_metrics"] = {
        "avg_sessions": round(avg_sessions, 1),
        "avg_page_views": round(avg_page_views, 1),
        "avg_cart_abandonment_rate": round(avg_cart_abandon, 2),
    }

    # 4. Churn risk analysis
    avg_churn_risk = sum(p.churn_risk_score for p in profiles) / n_profiles
    if avg_churn_risk >= 0.7:
        defining_traits.append("At-risk of churning")
    elif avg_churn_risk <= 0.3:
        defining_traits.append("Loyal customer base")

    trait_summary["churn_risk"] = {
        "avg_churn_score": round(avg_churn_risk, 2),
        "high_risk_count": sum(1 for p in profiles if p.churn_risk_score >= 0.7),
        "low_risk_count": sum(1 for p in profiles if p.churn_risk_score <= 0.3),
    }

    # 5. Device preference
    avg_mobile_ratio = sum(p.mobile_session_ratio for p in profiles) / n_profiles
    if avg_mobile_ratio >= 0.7:
        defining_traits.append("Mobile-first shoppers")
    elif avg_mobile_ratio <= 0.3:
        defining_traits.append("Desktop-preferred shoppers")
    else:
        defining_traits.append("Multi-device shoppers")

    trait_summary["device_preference"] = {
        "avg_mobile_ratio": round(avg_mobile_ratio, 2),
    }

    # 6. Temporal patterns
    preferred_days = [p.preferred_day_of_week for p in profiles if p.preferred_day_of_week is not None]
    preferred_hours = [p.preferred_hour_of_day for p in profiles if p.preferred_hour_of_day is not None]

    if preferred_days:
        day_counts = Counter(preferred_days)
        most_common_day = day_counts.most_common(1)[0][0]
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if most_common_day < len(day_names):
            if most_common_day >= 5:  # Weekend
                defining_traits.append("Weekend shoppers")
            else:
                defining_traits.append(f"{day_names[most_common_day]} peak activity")

        trait_summary["temporal_patterns"] = {
            "preferred_day": most_common_day,
            "day_distribution": dict(day_counts),
        }

    if preferred_hours:
        hour_counts = Counter(preferred_hours)
        most_common_hour = hour_counts.most_common(1)[0][0]
        if 6 <= most_common_hour < 12:
            defining_traits.append("Morning shoppers")
        elif 12 <= most_common_hour < 17:
            defining_traits.append("Afternoon shoppers")
        elif 17 <= most_common_hour < 21:
            defining_traits.append("Evening shoppers")
        else:
            defining_traits.append("Late-night shoppers")

        if "temporal_patterns" not in trait_summary:
            trait_summary["temporal_patterns"] = {}
        trait_summary["temporal_patterns"]["preferred_hour"] = most_common_hour
        trait_summary["temporal_patterns"]["hour_distribution"] = dict(hour_counts)

    return defining_traits, trait_summary


def create_segments_from_clusters(
    profiles: list[CustomerProfile],
    clustering_result: ClusteringResult,
    *,
    segment_id_prefix: str = "segment",
) -> list[Segment]:
    """
    Create Segment objects from clustering results.

    Args:
        profiles: List of customer profiles (same order as clustering)
        clustering_result: Results from cluster_customers
        segment_id_prefix: Prefix for segment IDs

    Returns:
        List of Segment objects
    """
    if len(profiles) != clustering_result.n_samples:
        raise ValueError(
            f"Profile count ({len(profiles)}) doesn't match clustering samples "
            f"({clustering_result.n_samples})"
        )

    # Group profiles by cluster
    cluster_profiles: dict[int, list[CustomerProfile]] = {
        i: [] for i in range(clustering_result.n_clusters)
    }

    for profile, label in zip(profiles, clustering_result.labels):
        cluster_profiles[int(label)].append(profile)

    segments: list[Segment] = []

    for cluster_id in range(clustering_result.n_clusters):
        profiles_in_cluster = cluster_profiles[cluster_id]

        if not profiles_in_cluster:
            continue

        # Calculate segment metrics
        total_clv = sum(
            (p.clv_estimate for p in profiles_in_cluster), Decimal("0")
        )
        avg_clv = total_clv / len(profiles_in_cluster)
        avg_aov = sum(
            (p.avg_order_value for p in profiles_in_cluster), Decimal("0")
        ) / len(profiles_in_cluster)

        # Extract defining traits from profiles
        defining_traits, trait_summary = _extract_segment_traits(profiles_in_cluster)

        # Add categorical modes to trait summary
        if cluster_id < len(clustering_result.categorical_centroids):
            cat_modes = clustering_result.categorical_centroids[cluster_id]
            trait_summary["categorical_modes"] = dict(zip(
                clustering_result.categorical_feature_names,
                cat_modes,
            ))

        # Create members
        members = [
            SegmentMember(
                internal_customer_id=p.internal_customer_id,
                membership_score=1.0,  # Hard clustering
            )
            for p in profiles_in_cluster
        ]

        # Get centroid as list (numerical only)
        centroid = clustering_result.centroids[cluster_id].tolist()

        # Generate descriptive name based on traits and categorical modes
        segment_name = f"Cluster {cluster_id}"
        if defining_traits:
            # Use first two distinctive traits for naming
            name_traits = [t for t in defining_traits[:2] if "customer" not in t.lower()]
            if name_traits:
                segment_name = f"{name_traits[0]}"

        segment = Segment(
            segment_id=f"{segment_id_prefix}_{cluster_id}",
            name=segment_name,
            description=f"Auto-generated cluster with {len(profiles_in_cluster)} customers. " +
                       ", ".join(defining_traits[:3]) if defining_traits else f"Auto-generated cluster with {len(profiles_in_cluster)} customers",
            members=members,
            size=len(profiles_in_cluster),
            total_clv=total_clv,
            avg_clv=avg_clv,
            avg_order_value=avg_aov,
            cluster_label=cluster_id,
            centroid=centroid,
            defining_traits=defining_traits,
            trait_summary=trait_summary,
        )

        segments.append(segment)

    return segments


class CustomerClusterer:
    """
    High-level interface for customer clustering using K-Prototypes.

    Combines feature extraction, standardization, clustering, and segment creation.
    Handles mixed data types (numerical + categorical) properly.
    """

    def __init__(
        self,
        *,
        n_clusters: int = 5,
        random_seed: int = 42,
        auto_select_k: bool = False,
        k_range: tuple[int, int] = (2, 10),
        gamma: float | None = None,
    ) -> None:
        """
        Initialize CustomerClusterer.

        Args:
            n_clusters: Number of clusters (ignored if auto_select_k=True)
            random_seed: Random seed for reproducibility
            auto_select_k: Automatically select optimal k
            k_range: Range of k to test if auto_select_k=True
            gamma: Weight for categorical features (None = auto-calculate)
        """
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.auto_select_k = auto_select_k
        self.k_range = k_range
        self.gamma = gamma

        self._scaler: StandardScaler | None = None
        self._last_result: ClusteringResult | None = None

    def fit_predict(
        self,
        profiles: list[CustomerProfile],
    ) -> ClusteringResult:
        """
        Extract features, cluster, and return results.

        Args:
            profiles: List of customer profiles

        Returns:
            ClusteringResult with cluster assignments
        """
        # Extract mixed features
        features = extract_mixed_features(profiles)

        # Standardize numerical features only
        standardized = standardize_numerical_features(features)
        self._scaler = standardized.scaler

        # Auto-select k if requested
        n_clusters = self.n_clusters
        if self.auto_select_k:
            k_analysis = find_optimal_k(
                standardized,
                k_range=self.k_range,
                random_seed=self.random_seed,
            )
            n_clusters = k_analysis["optimal_k"]

        # Cluster using K-Prototypes
        result = cluster_customers(
            standardized,
            n_clusters=n_clusters,
            random_seed=self.random_seed,
            gamma=self.gamma,
        )

        self._last_result = result
        return result

    def create_segments(
        self,
        profiles: list[CustomerProfile],
        *,
        segment_id_prefix: str = "segment",
    ) -> list[Segment]:
        """
        Cluster profiles and create segments.

        Args:
            profiles: List of customer profiles
            segment_id_prefix: Prefix for segment IDs

        Returns:
            List of Segment objects
        """
        result = self.fit_predict(profiles)
        return create_segments_from_clusters(
            profiles,
            result,
            segment_id_prefix=segment_id_prefix,
        )

    @property
    def last_result(self) -> ClusteringResult | None:
        """Get the last clustering result."""
        return self._last_result


def get_cluster_summary(result: ClusteringResult) -> dict[str, Any]:
    """
    Generate a human-readable summary of clustering results.

    Args:
        result: ClusteringResult from clustering

    Returns:
        Dictionary with summary information
    """
    summary: dict[str, Any] = {
        "algorithm": "K-Prototypes",
        "n_clusters": result.n_clusters,
        "n_samples": result.n_samples,
        "cost": result.cost,
        "silhouette_score": result.silhouette,
        "cluster_sizes": result.cluster_sizes,
        "numerical_features": result.feature_names,
        "categorical_features": result.categorical_feature_names,
    }

    # Size distribution
    sizes = list(result.cluster_sizes.values())
    summary["size_stats"] = {
        "min": min(sizes),
        "max": max(sizes),
        "mean": sum(sizes) / len(sizes),
        "std": float(np.std(sizes)),
    }

    # Categorical modes for each cluster
    summary["cluster_modes"] = {}
    for i, modes in enumerate(result.categorical_centroids):
        summary["cluster_modes"][i] = dict(zip(
            result.categorical_feature_names,
            modes,
        ))

    return summary


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Keep old names for backward compatibility
FeatureMatrix = MixedFeatureMatrix
extract_features = extract_mixed_features
standardize_features = standardize_numerical_features
