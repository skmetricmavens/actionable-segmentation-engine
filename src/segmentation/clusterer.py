"""
Module: clusterer

Purpose: ML-based customer clustering for segmentation.

Uses scikit-learn for clustering with deterministic seeding.
Produces segment candidates that can be evaluated by LLM for actionability.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
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
    centroids: NDArray[np.float64]
    inertia: float
    silhouette: float | None
    n_clusters: int
    n_samples: int
    feature_names: list[str]

    # Cluster-level statistics
    cluster_sizes: dict[int, int] = field(default_factory=dict)
    cluster_stats: dict[int, dict[str, float]] = field(default_factory=dict)


@dataclass
class FeatureMatrix:
    """Container for feature extraction results."""

    matrix: NDArray[np.float64]
    feature_names: list[str]
    customer_ids: list[str]
    scaler: StandardScaler | None = None


def extract_features(
    profiles: list[CustomerProfile],
    *,
    include_value_features: bool = True,
    include_engagement_features: bool = True,
    include_temporal_features: bool = True,
    include_channel_features: bool = True,
) -> FeatureMatrix:
    """
    Extract numerical features from customer profiles for clustering.

    Args:
        profiles: List of customer profiles
        include_value_features: Include revenue/CLV features
        include_engagement_features: Include session/view features
        include_temporal_features: Include day/hour features
        include_channel_features: Include device features

    Returns:
        FeatureMatrix with extracted features

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

    feature_names: list[str] = []
    customer_ids: list[str] = []
    feature_rows: list[list[float]] = []

    for profile in profiles:
        customer_ids.append(profile.internal_customer_id)
        row: list[float] = []

        # Value features
        if include_value_features:
            if "total_revenue" not in feature_names:
                feature_names.extend([
                    "total_revenue",
                    "avg_order_value",
                    "clv_estimate",
                    "total_purchases",
                    "purchase_frequency",
                    "churn_risk",
                ])

            row.extend([
                float(profile.total_revenue),
                float(profile.avg_order_value),
                float(profile.clv_estimate),
                float(profile.total_purchases),
                profile.purchase_frequency_per_month,
                profile.churn_risk_score,
            ])

        # Engagement features
        if include_engagement_features:
            if "total_sessions" not in feature_names:
                feature_names.extend([
                    "total_sessions",
                    "total_page_views",
                    "total_items_viewed",
                    "cart_abandonment_rate",
                ])

            row.extend([
                float(profile.total_sessions),
                float(profile.total_page_views),
                float(profile.total_items_viewed),
                profile.cart_abandonment_rate,
            ])

        # Temporal features
        if include_temporal_features:
            if "preferred_day" not in feature_names:
                feature_names.extend([
                    "preferred_day",
                    "preferred_hour",
                ])

            # Use -1 for missing temporal data
            day = profile.preferred_day_of_week if profile.preferred_day_of_week is not None else -1
            hour = profile.preferred_hour_of_day if profile.preferred_hour_of_day is not None else -1

            row.extend([float(day), float(hour)])

        # Channel features
        if include_channel_features:
            if "mobile_ratio" not in feature_names:
                feature_names.append("mobile_ratio")

            row.append(profile.mobile_session_ratio)

        feature_rows.append(row)

    matrix = np.array(feature_rows, dtype=np.float64)

    return FeatureMatrix(
        matrix=matrix,
        feature_names=feature_names,
        customer_ids=customer_ids,
    )


def standardize_features(
    feature_matrix: FeatureMatrix,
    *,
    scaler: StandardScaler | None = None,
) -> FeatureMatrix:
    """
    Standardize features using StandardScaler.

    Args:
        feature_matrix: FeatureMatrix to standardize
        scaler: Optional pre-fitted scaler (for applying to new data)

    Returns:
        New FeatureMatrix with standardized values
    """
    if scaler is None:
        scaler = StandardScaler()
        standardized = scaler.fit_transform(feature_matrix.matrix)
    else:
        standardized = scaler.transform(feature_matrix.matrix)

    return FeatureMatrix(
        matrix=standardized,
        feature_names=feature_matrix.feature_names,
        customer_ids=feature_matrix.customer_ids,
        scaler=scaler,
    )


def cluster_customers(
    feature_matrix: FeatureMatrix,
    *,
    n_clusters: int = 5,
    random_seed: int = 42,
    max_iter: int = 300,
    n_init: int = 10,
) -> ClusteringResult:
    """
    Perform KMeans clustering on customer features.

    Args:
        feature_matrix: Standardized feature matrix
        n_clusters: Number of clusters to create
        random_seed: Random seed for reproducibility
        max_iter: Maximum iterations for KMeans
        n_init: Number of initializations

    Returns:
        ClusteringResult with cluster assignments and statistics

    Raises:
        ClusteringError: If clustering fails
        InsufficientDataError: If insufficient samples
    """
    n_samples = feature_matrix.matrix.shape[0]

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
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            max_iter=max_iter,
            n_init=n_init,
        )
        labels = kmeans.fit_predict(feature_matrix.matrix)
        centroids = kmeans.cluster_centers_

        # Calculate silhouette score if we have enough samples
        silhouette = None
        if n_samples >= n_clusters + 1 and n_clusters > 1:
            silhouette = float(silhouette_score(feature_matrix.matrix, labels))

        # Calculate cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(label): int(count) for label, count in zip(unique_labels, counts)}

        # Calculate cluster statistics
        cluster_stats: dict[int, dict[str, float]] = {}
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_features = feature_matrix.matrix[cluster_mask]

            stats: dict[str, float] = {}
            for i, feature_name in enumerate(feature_matrix.feature_names):
                stats[f"{feature_name}_mean"] = float(np.mean(cluster_features[:, i]))
                stats[f"{feature_name}_std"] = float(np.std(cluster_features[:, i]))

            cluster_stats[int(cluster_id)] = stats

        return ClusteringResult(
            labels=labels,
            centroids=centroids,
            inertia=kmeans.inertia_,
            silhouette=silhouette,
            n_clusters=n_clusters,
            n_samples=n_samples,
            feature_names=feature_matrix.feature_names,
            cluster_sizes=cluster_sizes,
            cluster_stats=cluster_stats,
        )

    except Exception as e:
        raise ClusteringError(
            f"Clustering failed: {e!s}",
            n_clusters=n_clusters,
            n_samples=n_samples,
        ) from e


def find_optimal_k(
    feature_matrix: FeatureMatrix,
    *,
    k_range: tuple[int, int] = (2, 10),
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Find optimal number of clusters using elbow method and silhouette scores.

    Args:
        feature_matrix: Standardized feature matrix
        k_range: Range of k values to test (min, max)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with analysis results
    """
    min_k, max_k = k_range
    n_samples = feature_matrix.matrix.shape[0]

    # Adjust max_k if needed
    max_k = min(max_k, n_samples - 1)
    min_k = max(2, min_k)

    if min_k > max_k:
        return {
            "optimal_k": min_k,
            "inertias": {},
            "silhouettes": {},
            "reason": "insufficient_samples",
        }

    inertias: dict[int, float] = {}
    silhouettes: dict[int, float] = {}

    for k in range(min_k, max_k + 1):
        try:
            result = cluster_customers(
                feature_matrix,
                n_clusters=k,
                random_seed=random_seed,
            )
            inertias[k] = result.inertia
            if result.silhouette is not None:
                silhouettes[k] = result.silhouette
        except (ClusteringError, InsufficientDataError):
            continue

    # Find optimal k by silhouette score
    optimal_k = min_k
    if silhouettes:
        optimal_k = max(silhouettes.keys(), key=lambda k: silhouettes[k])

    return {
        "optimal_k": optimal_k,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "reason": "silhouette_score" if silhouettes else "default",
    }


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

        # Create members
        members = [
            SegmentMember(
                internal_customer_id=p.internal_customer_id,
                membership_score=1.0,  # Hard clustering
            )
            for p in profiles_in_cluster
        ]

        # Get centroid as list
        centroid = clustering_result.centroids[cluster_id].tolist()

        segment = Segment(
            segment_id=f"{segment_id_prefix}_{cluster_id}",
            name=f"Cluster {cluster_id}",
            description=f"Auto-generated cluster with {len(profiles_in_cluster)} customers",
            members=members,
            size=len(profiles_in_cluster),
            total_clv=total_clv,
            avg_clv=avg_clv,
            avg_order_value=avg_aov,
            cluster_label=cluster_id,
            centroid=centroid,
        )

        segments.append(segment)

    return segments


class CustomerClusterer:
    """
    High-level interface for customer clustering.

    Combines feature extraction, standardization, clustering, and segment creation.
    """

    def __init__(
        self,
        *,
        n_clusters: int = 5,
        random_seed: int = 42,
        auto_select_k: bool = False,
        k_range: tuple[int, int] = (2, 10),
    ) -> None:
        """
        Initialize CustomerClusterer.

        Args:
            n_clusters: Number of clusters (ignored if auto_select_k=True)
            random_seed: Random seed for reproducibility
            auto_select_k: Automatically select optimal k
            k_range: Range of k to test if auto_select_k=True
        """
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.auto_select_k = auto_select_k
        self.k_range = k_range

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
        # Extract features
        features = extract_features(profiles)

        # Standardize
        standardized = standardize_features(features)
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

        # Cluster
        result = cluster_customers(
            standardized,
            n_clusters=n_clusters,
            random_seed=self.random_seed,
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
        "n_clusters": result.n_clusters,
        "n_samples": result.n_samples,
        "inertia": result.inertia,
        "silhouette_score": result.silhouette,
        "cluster_sizes": result.cluster_sizes,
    }

    # Size distribution
    sizes = list(result.cluster_sizes.values())
    summary["size_stats"] = {
        "min": min(sizes),
        "max": max(sizes),
        "mean": sum(sizes) / len(sizes),
        "std": float(np.std(sizes)),
    }

    return summary
