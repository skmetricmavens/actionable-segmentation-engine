"""
Tests for ML clustering module.
"""

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from src.data.schemas import CustomerProfile, Segment
from src.exceptions import ClusteringError, InsufficientDataError
from src.segmentation.clusterer import (
    ClusteringResult,
    CustomerClusterer,
    FeatureMatrix,
    cluster_customers,
    create_segments_from_clusters,
    extract_features,
    find_optimal_k,
    get_cluster_summary,
    standardize_features,
)


# =============================================================================
# FIXTURES
# =============================================================================


def make_profile(
    customer_id: str,
    *,
    total_revenue: float = 100.0,
    clv_estimate: float = 300.0,
    churn_risk_score: float = 0.3,
    total_purchases: int = 3,
    total_sessions: int = 5,
    mobile_session_ratio: float = 0.3,
) -> CustomerProfile:
    """Helper to create CustomerProfile for testing."""
    return CustomerProfile(
        internal_customer_id=customer_id,
        first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
        last_seen=datetime(2024, 3, 31, tzinfo=timezone.utc),
        total_purchases=total_purchases,
        total_revenue=Decimal(str(total_revenue)),
        avg_order_value=Decimal(str(total_revenue / max(total_purchases, 1))),
        purchase_frequency_per_month=float(total_purchases) / 3.0,
        total_sessions=total_sessions,
        total_page_views=total_sessions * 3,
        total_items_viewed=total_sessions * 2,
        total_cart_additions=total_purchases + 2,
        cart_abandonment_rate=0.4,
        preferred_day_of_week=1,
        preferred_hour_of_day=14,
        mobile_session_ratio=mobile_session_ratio,
        clv_estimate=Decimal(str(clv_estimate)),
        churn_risk_score=churn_risk_score,
    )


def make_diverse_profiles(n: int = 20) -> list[CustomerProfile]:
    """Create diverse profiles for clustering tests."""
    profiles = []

    # High-value customers (cluster 1)
    for i in range(n // 4):
        profiles.append(
            make_profile(
                f"high_value_{i}",
                total_revenue=500.0 + i * 50,
                clv_estimate=1000.0 + i * 100,
                total_purchases=10 + i,
                total_sessions=15 + i,
                mobile_session_ratio=0.2,
            )
        )

    # Low-value browsing (cluster 2)
    for i in range(n // 4):
        profiles.append(
            make_profile(
                f"browser_{i}",
                total_revenue=20.0 + i * 5,
                clv_estimate=50.0 + i * 10,
                total_purchases=1,
                total_sessions=20 + i,
                mobile_session_ratio=0.8,
            )
        )

    # Moderate customers (cluster 3)
    for i in range(n // 4):
        profiles.append(
            make_profile(
                f"moderate_{i}",
                total_revenue=150.0 + i * 20,
                clv_estimate=300.0 + i * 30,
                total_purchases=5 + i,
                total_sessions=8 + i,
                mobile_session_ratio=0.5,
            )
        )

    # Churning customers (cluster 4)
    for i in range(n // 4):
        profiles.append(
            make_profile(
                f"churning_{i}",
                total_revenue=100.0 + i * 10,
                clv_estimate=100.0 + i * 10,
                total_purchases=2,
                total_sessions=2,
                churn_risk_score=0.8 + i * 0.01,
                mobile_session_ratio=0.3,
            )
        )

    return profiles


# =============================================================================
# FEATURE EXTRACTION TESTS
# =============================================================================


class TestExtractFeatures:
    """Tests for extract_features function."""

    def test_empty_profiles_raises(self) -> None:
        """Test that empty profiles raises error."""
        with pytest.raises(InsufficientDataError):
            extract_features([])

    def test_single_profile(self) -> None:
        """Test feature extraction from single profile."""
        profile = make_profile("cust_1")
        result = extract_features([profile])

        assert isinstance(result, FeatureMatrix)
        assert result.matrix.shape[0] == 1
        assert len(result.customer_ids) == 1
        assert result.customer_ids[0] == "cust_1"

    def test_multiple_profiles(self) -> None:
        """Test feature extraction from multiple profiles."""
        profiles = [make_profile(f"cust_{i}") for i in range(5)]
        result = extract_features(profiles)

        assert result.matrix.shape[0] == 5
        assert len(result.customer_ids) == 5

    def test_all_feature_groups(self) -> None:
        """Test all feature groups are included."""
        profile = make_profile("cust_1")
        result = extract_features([profile])

        # Should have value, engagement, temporal, and channel features
        assert "total_revenue" in result.feature_names
        assert "total_sessions" in result.feature_names
        assert "preferred_day" in result.feature_names
        assert "mobile_ratio" in result.feature_names

    def test_selective_features(self) -> None:
        """Test selective feature extraction."""
        profile = make_profile("cust_1")
        result = extract_features(
            [profile],
            include_value_features=True,
            include_engagement_features=False,
            include_temporal_features=False,
            include_channel_features=False,
        )

        assert "total_revenue" in result.feature_names
        assert "total_sessions" not in result.feature_names
        assert "preferred_day" not in result.feature_names
        assert "mobile_ratio" not in result.feature_names


# =============================================================================
# STANDARDIZATION TESTS
# =============================================================================


class TestStandardizeFeatures:
    """Tests for standardize_features function."""

    def test_standardization(self) -> None:
        """Test that features are standardized."""
        profiles = make_diverse_profiles(20)
        # Exclude temporal features which may have constant values
        features = extract_features(
            profiles,
            include_temporal_features=False,
        )
        standardized = standardize_features(features)

        # Mean should be close to 0
        means = np.mean(standardized.matrix, axis=0)
        assert np.allclose(means, 0, atol=0.1)

        # Std should be 1 for non-constant features (std=0 for constant features is expected)
        stds = np.std(standardized.matrix, axis=0)
        # Check that most features have std close to 1 (constant features will have std=0)
        non_constant_mask = stds > 0.5
        assert np.sum(non_constant_mask) >= len(stds) - 2  # At most 2 constant features
        assert np.allclose(stds[non_constant_mask], 1, atol=0.1)

    def test_scaler_preserved(self) -> None:
        """Test that scaler is preserved."""
        profiles = make_diverse_profiles(10)
        features = extract_features(profiles)
        standardized = standardize_features(features)

        assert standardized.scaler is not None

    def test_reuse_scaler(self) -> None:
        """Test reusing scaler for new data."""
        profiles1 = make_diverse_profiles(10)
        profiles2 = make_diverse_profiles(10)

        features1 = extract_features(profiles1)
        standardized1 = standardize_features(features1)

        features2 = extract_features(profiles2)
        standardized2 = standardize_features(features2, scaler=standardized1.scaler)

        # Shapes should match
        assert standardized1.matrix.shape[1] == standardized2.matrix.shape[1]


# =============================================================================
# CLUSTERING TESTS
# =============================================================================


class TestClusterCustomers:
    """Tests for cluster_customers function."""

    def test_insufficient_samples(self) -> None:
        """Test error when samples < clusters."""
        profiles = [make_profile(f"cust_{i}") for i in range(3)]
        features = standardize_features(extract_features(profiles))

        with pytest.raises(InsufficientDataError):
            cluster_customers(features, n_clusters=5)

    def test_basic_clustering(self) -> None:
        """Test basic clustering works."""
        profiles = make_diverse_profiles(20)
        features = standardize_features(extract_features(profiles))

        result = cluster_customers(features, n_clusters=4)

        assert isinstance(result, ClusteringResult)
        assert result.n_clusters == 4
        assert result.n_samples == 20
        assert len(result.labels) == 20

    def test_deterministic_results(self) -> None:
        """Test clustering is deterministic with same seed."""
        profiles = make_diverse_profiles(20)
        features = standardize_features(extract_features(profiles))

        result1 = cluster_customers(features, n_clusters=4, random_seed=42)
        result2 = cluster_customers(features, n_clusters=4, random_seed=42)

        assert np.array_equal(result1.labels, result2.labels)

    def test_cluster_sizes(self) -> None:
        """Test cluster sizes are computed."""
        profiles = make_diverse_profiles(20)
        features = standardize_features(extract_features(profiles))

        result = cluster_customers(features, n_clusters=4)

        assert len(result.cluster_sizes) == 4
        assert sum(result.cluster_sizes.values()) == 20

    def test_cluster_stats(self) -> None:
        """Test cluster statistics are computed."""
        profiles = make_diverse_profiles(20)
        features = standardize_features(extract_features(profiles))

        result = cluster_customers(features, n_clusters=4)

        assert len(result.cluster_stats) == 4
        for cluster_id, stats in result.cluster_stats.items():
            assert f"{features.feature_names[0]}_mean" in stats
            assert f"{features.feature_names[0]}_std" in stats

    def test_silhouette_score(self) -> None:
        """Test silhouette score is computed."""
        profiles = make_diverse_profiles(20)
        features = standardize_features(extract_features(profiles))

        result = cluster_customers(features, n_clusters=4)

        assert result.silhouette is not None
        assert -1 <= result.silhouette <= 1


# =============================================================================
# OPTIMAL K TESTS
# =============================================================================


class TestFindOptimalK:
    """Tests for find_optimal_k function."""

    def test_optimal_k_selection(self) -> None:
        """Test optimal k is selected."""
        profiles = make_diverse_profiles(40)
        features = standardize_features(extract_features(profiles))

        analysis = find_optimal_k(features, k_range=(2, 6))

        assert "optimal_k" in analysis
        assert 2 <= analysis["optimal_k"] <= 6
        assert len(analysis["silhouettes"]) > 0

    def test_small_sample_handling(self) -> None:
        """Test handling of small sample count adjusts k_range."""
        profiles = [make_profile(f"cust_{i}") for i in range(3)]
        features = standardize_features(extract_features(profiles))

        # With 3 samples, max_k will be adjusted to 2
        analysis = find_optimal_k(features, k_range=(2, 10))

        # Should still return a result, adjusted for sample size
        assert "optimal_k" in analysis
        assert analysis["optimal_k"] >= 2


# =============================================================================
# SEGMENT CREATION TESTS
# =============================================================================


class TestCreateSegmentsFromClusters:
    """Tests for create_segments_from_clusters function."""

    def test_segment_creation(self) -> None:
        """Test segments are created from clusters."""
        profiles = make_diverse_profiles(20)
        features = standardize_features(extract_features(profiles))
        result = cluster_customers(features, n_clusters=4)

        segments = create_segments_from_clusters(profiles, result)

        assert len(segments) == 4
        assert all(isinstance(s, Segment) for s in segments)

    def test_segment_metrics(self) -> None:
        """Test segment metrics are calculated."""
        profiles = make_diverse_profiles(20)
        features = standardize_features(extract_features(profiles))
        result = cluster_customers(features, n_clusters=4)

        segments = create_segments_from_clusters(profiles, result)

        for segment in segments:
            assert segment.size > 0
            assert segment.total_clv >= 0
            assert segment.avg_clv >= 0
            assert len(segment.members) == segment.size

    def test_segment_id_prefix(self) -> None:
        """Test custom segment ID prefix."""
        profiles = make_diverse_profiles(10)
        features = standardize_features(extract_features(profiles))
        result = cluster_customers(features, n_clusters=2)

        segments = create_segments_from_clusters(
            profiles, result, segment_id_prefix="custom"
        )

        assert all(s.segment_id.startswith("custom_") for s in segments)

    def test_mismatch_raises(self) -> None:
        """Test error on profile count mismatch."""
        profiles = make_diverse_profiles(20)
        features = standardize_features(extract_features(profiles))
        result = cluster_customers(features, n_clusters=4)

        # Try with wrong number of profiles
        with pytest.raises(ValueError):
            create_segments_from_clusters(profiles[:10], result)


# =============================================================================
# CUSTOMER CLUSTERER CLASS TESTS
# =============================================================================


class TestCustomerClusterer:
    """Tests for CustomerClusterer class."""

    def test_fit_predict(self) -> None:
        """Test fit_predict method."""
        profiles = make_diverse_profiles(20)
        clusterer = CustomerClusterer(n_clusters=4)

        result = clusterer.fit_predict(profiles)

        assert isinstance(result, ClusteringResult)
        assert clusterer.last_result is not None

    def test_create_segments(self) -> None:
        """Test create_segments method."""
        profiles = make_diverse_profiles(20)
        clusterer = CustomerClusterer(n_clusters=4)

        segments = clusterer.create_segments(profiles)

        assert len(segments) == 4
        assert all(isinstance(s, Segment) for s in segments)

    def test_auto_select_k(self) -> None:
        """Test automatic k selection."""
        profiles = make_diverse_profiles(40)
        clusterer = CustomerClusterer(auto_select_k=True, k_range=(2, 6))

        result = clusterer.fit_predict(profiles)

        assert 2 <= result.n_clusters <= 6


# =============================================================================
# CLUSTER SUMMARY TESTS
# =============================================================================


class TestGetClusterSummary:
    """Tests for get_cluster_summary function."""

    def test_summary_contents(self) -> None:
        """Test summary contains expected fields."""
        profiles = make_diverse_profiles(20)
        features = standardize_features(extract_features(profiles))
        result = cluster_customers(features, n_clusters=4)

        summary = get_cluster_summary(result)

        assert "n_clusters" in summary
        assert "n_samples" in summary
        assert "inertia" in summary
        assert "silhouette_score" in summary
        assert "cluster_sizes" in summary
        assert "size_stats" in summary


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegrationWithSyntheticData:
    """Integration tests with synthetic data."""

    def test_end_to_end_clustering(self) -> None:
        """Test complete clustering pipeline with synthetic data."""
        from src.data.joiner import resolve_customer_merges
        from src.data.synthetic_generator import SyntheticDataGenerator
        from src.features.profile_builder import build_profiles_batch

        # Generate synthetic data
        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(
            n_customers=50,
            date_range=date_range,
        )

        # Build profiles
        merge_map = resolve_customer_merges(dataset.id_history)
        ref_date = datetime(2024, 4, 1, tzinfo=timezone.utc)
        profiles = build_profiles_batch(
            dataset.events,
            merge_map=merge_map,
            reference_date=ref_date,
        )

        # Cluster
        clusterer = CustomerClusterer(n_clusters=4)
        segments = clusterer.create_segments(profiles)

        # Verify
        assert len(segments) == 4
        total_members = sum(s.size for s in segments)
        assert total_members == len(profiles)

        # Summary
        summary = get_cluster_summary(clusterer.last_result)
        assert summary["n_clusters"] == 4
