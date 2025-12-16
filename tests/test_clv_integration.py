"""
Tests for CLV predictor integration in whitespace and clustering modules.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.schemas import CategoryAffinity, CustomerProfile
from src.features.clv.predictor import CLVPrediction
from src.segmentation.clusterer import (
    ClusteringResult,
    create_segments_from_clusters,
)
from src.segmentation.whitespace import WhitespaceAnalyzer


# =============================================================================
# FIXTURES
# =============================================================================


def make_profile(
    customer_id: str,
    *,
    total_revenue: float = 100.0,
    total_purchases: int = 3,
    total_sessions: int = 5,
    category_affinities: list[CategoryAffinity] | None = None,
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
        mobile_session_ratio=0.3,
        clv_estimate=Decimal(str(total_revenue * 3)),
        churn_risk_score=0.3,
        category_affinities=category_affinities or [],
    )


def make_mock_predictor(predictions: dict[str, float]) -> MagicMock:
    """Create a mock CLVPredictor that returns specified predictions."""
    predictor = MagicMock()

    def predict_impl(profile, n_top_features=5):
        clv = predictions.get(profile.internal_customer_id, 100.0)
        return CLVPrediction(
            customer_id=profile.internal_customer_id,
            predicted_clv=Decimal(str(clv)),
            confidence=0.85,
            prediction_timestamp=datetime.now(),
            model_version="test_v1",
            top_features=[("recency", 0.3), ("frequency", 0.25)],
        )

    def predict_batch_impl(profiles, n_top_features=5):
        return [predict_impl(p, n_top_features) for p in profiles]

    predictor.predict.side_effect = predict_impl
    predictor.predict_batch.side_effect = predict_batch_impl

    return predictor


# =============================================================================
# WHITESPACE ANALYZER TESTS
# =============================================================================


class TestWhitespaceAnalyzerCLVIntegration:
    """Test CLV predictor integration in WhitespaceAnalyzer."""

    def test_analyzer_without_predictor_uses_total_revenue(self):
        """When no predictor is provided, should use total_revenue as CLV."""
        # Create profiles with known revenues
        profiles = [
            make_profile(
                f"buyer_{i}",
                total_revenue=100.0 * (i + 1),
                category_affinities=[
                    CategoryAffinity(
                        category="Electronics",
                        view_count=10,
                        purchase_count=2,
                        engagement_score=0.8,
                    )
                ],
            )
            for i in range(5)
        ]

        # Add a candidate (engaged but no purchase)
        profiles.append(
            make_profile(
                "candidate_1",
                total_revenue=50.0,
                category_affinities=[
                    CategoryAffinity(
                        category="Electronics",
                        view_count=15,
                        purchase_count=0,
                        engagement_score=0.7,
                    )
                ],
            )
        )

        analyzer = WhitespaceAnalyzer(min_seed_size=3, clv_predictor=None)

        # _get_avg_clv should return average of total_revenue
        buyers = profiles[:5]
        avg_clv = analyzer._get_avg_clv(buyers)

        expected_avg = sum(p.total_revenue for p in buyers) / len(buyers)
        assert avg_clv == expected_avg

    def test_analyzer_with_predictor_uses_ml_predictions(self):
        """When predictor is provided, should use ML predictions for CLV."""
        # Create profiles
        profiles = [
            make_profile(
                f"buyer_{i}",
                total_revenue=100.0,  # Same revenue for all
                category_affinities=[
                    CategoryAffinity(
                        category="Electronics",
                        view_count=10,
                        purchase_count=2,
                        engagement_score=0.8,
                    )
                ],
            )
            for i in range(5)
        ]

        # Mock predictor returns different values than total_revenue
        mock_predictions = {
            "buyer_0": 500.0,
            "buyer_1": 600.0,
            "buyer_2": 700.0,
            "buyer_3": 800.0,
            "buyer_4": 900.0,
        }
        mock_predictor = make_mock_predictor(mock_predictions)

        analyzer = WhitespaceAnalyzer(min_seed_size=3, clv_predictor=mock_predictor)

        # _get_avg_clv should return average of ML predictions, not total_revenue
        avg_clv = analyzer._get_avg_clv(profiles)

        expected_avg = Decimal(str(sum(mock_predictions.values()) / len(mock_predictions)))
        assert avg_clv == expected_avg

        # Verify predictor was called
        mock_predictor.predict_batch.assert_called_once()

    def test_analyzer_caches_predictions(self):
        """Predictions should be cached to avoid redundant calls."""
        profiles = [
            make_profile(
                "buyer_1",
                total_revenue=100.0,
                category_affinities=[
                    CategoryAffinity(
                        category="Electronics",
                        view_count=10,
                        purchase_count=2,
                        engagement_score=0.8,
                    )
                ],
            )
        ]

        mock_predictor = make_mock_predictor({"buyer_1": 500.0})
        analyzer = WhitespaceAnalyzer(min_seed_size=1, clv_predictor=mock_predictor)

        # Call twice with same profile
        analyzer._get_avg_clv(profiles)
        analyzer._get_avg_clv(profiles)

        # Should only call predict_batch once due to caching
        assert mock_predictor.predict_batch.call_count == 1

    def test_get_clv_single_profile(self):
        """Test _get_clv for a single profile."""
        profile = make_profile("customer_1", total_revenue=100.0)
        mock_predictor = make_mock_predictor({"customer_1": 999.0})

        analyzer = WhitespaceAnalyzer(clv_predictor=mock_predictor)

        clv = analyzer._get_clv(profile)

        assert clv == Decimal("999.0")

    def test_get_clv_without_predictor_returns_total_revenue(self):
        """Without predictor, _get_clv should return total_revenue."""
        profile = make_profile("customer_1", total_revenue=123.45)

        analyzer = WhitespaceAnalyzer(clv_predictor=None)

        clv = analyzer._get_clv(profile)

        assert clv == Decimal("123.45")


# =============================================================================
# CLUSTERER TESTS
# =============================================================================


class TestCreateSegmentsFromClustersCLVIntegration:
    """Test CLV predictor integration in create_segments_from_clusters."""

    def test_without_predictor_uses_total_revenue(self):
        """When no predictor, segments should use total_revenue for CLV."""
        profiles = [
            make_profile(f"customer_{i}", total_revenue=100.0 * (i + 1))
            for i in range(6)
        ]

        # Create a simple 2-cluster result
        clustering_result = ClusteringResult(
            labels=np.array([0, 0, 0, 1, 1, 1]),
            centroids=np.array([[1, 2], [3, 4]]),
            inertia=100.0,
            silhouette=0.5,
            n_clusters=2,
            n_samples=6,
            feature_names=["f1", "f2"],
        )

        segments = create_segments_from_clusters(
            profiles,
            clustering_result,
            clv_predictor=None,
        )

        assert len(segments) == 2

        # Cluster 0: customers 0, 1, 2 with revenues 100, 200, 300
        # Total CLV should be 600, avg should be 200
        cluster_0_segment = next(s for s in segments if s.cluster_label == 0)
        assert cluster_0_segment.total_clv == Decimal("600")
        assert cluster_0_segment.avg_clv == Decimal("200")

        # Cluster 1: customers 3, 4, 5 with revenues 400, 500, 600
        # Total CLV should be 1500, avg should be 500
        cluster_1_segment = next(s for s in segments if s.cluster_label == 1)
        assert cluster_1_segment.total_clv == Decimal("1500")
        assert cluster_1_segment.avg_clv == Decimal("500")

    def test_with_predictor_uses_ml_predictions(self):
        """When predictor provided, segments should use ML predictions for CLV."""
        profiles = [
            make_profile(f"customer_{i}", total_revenue=100.0)  # Same revenue
            for i in range(6)
        ]

        # Mock predictor returns different values
        mock_predictions = {
            "customer_0": 1000.0,
            "customer_1": 2000.0,
            "customer_2": 3000.0,
            "customer_3": 4000.0,
            "customer_4": 5000.0,
            "customer_5": 6000.0,
        }
        mock_predictor = make_mock_predictor(mock_predictions)

        clustering_result = ClusteringResult(
            labels=np.array([0, 0, 0, 1, 1, 1]),
            centroids=np.array([[1, 2], [3, 4]]),
            inertia=100.0,
            silhouette=0.5,
            n_clusters=2,
            n_samples=6,
            feature_names=["f1", "f2"],
        )

        segments = create_segments_from_clusters(
            profiles,
            clustering_result,
            clv_predictor=mock_predictor,
        )

        assert len(segments) == 2

        # Cluster 0: predictions 1000, 2000, 3000 = 6000 total, 2000 avg
        cluster_0_segment = next(s for s in segments if s.cluster_label == 0)
        assert cluster_0_segment.total_clv == Decimal("6000")
        assert cluster_0_segment.avg_clv == Decimal("2000")

        # Cluster 1: predictions 4000, 5000, 6000 = 15000 total, 5000 avg
        cluster_1_segment = next(s for s in segments if s.cluster_label == 1)
        assert cluster_1_segment.total_clv == Decimal("15000")
        assert cluster_1_segment.avg_clv == Decimal("5000")

        # Verify predictor was called
        mock_predictor.predict_batch.assert_called_once()

    def test_predictor_called_once_for_all_profiles(self):
        """Should batch predict all profiles at once for efficiency."""
        profiles = [make_profile(f"c_{i}") for i in range(10)]
        mock_predictor = make_mock_predictor({f"c_{i}": 100.0 for i in range(10)})

        clustering_result = ClusteringResult(
            labels=np.array([0] * 5 + [1] * 5),
            centroids=np.array([[1], [2]]),
            inertia=50.0,
            silhouette=0.6,
            n_clusters=2,
            n_samples=10,
            feature_names=["f1"],
        )

        create_segments_from_clusters(
            profiles,
            clustering_result,
            clv_predictor=mock_predictor,
        )

        # Should call predict_batch exactly once with all profiles
        assert mock_predictor.predict_batch.call_count == 1
        call_args = mock_predictor.predict_batch.call_args[0][0]
        assert len(call_args) == 10


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestEndToEndCLVIntegration:
    """End-to-end tests for CLV predictor integration."""

    def test_whitespace_analysis_with_predictor_no_candidates(self):
        """Test whitespace analysis with no candidates (no opportunities reported)."""
        # Create only buyers (no candidates = no lookalikes = no opportunities)
        buyers = [
            make_profile(
                f"buyer_{i}",
                total_revenue=100.0,
                total_purchases=5,
                total_sessions=10,
                category_affinities=[
                    CategoryAffinity(
                        category="Electronics",
                        view_count=20,
                        purchase_count=3,
                        engagement_score=0.9,
                    )
                ],
            )
            for i in range(10)
        ]

        # Mock predictor: returns higher CLV than total_revenue
        predictions = {f"buyer_{i}": 500.0 for i in range(10)}
        mock_predictor = make_mock_predictor(predictions)

        analyzer = WhitespaceAnalyzer(
            min_seed_size=5,
            clv_predictor=mock_predictor,
        )

        result = analyzer.analyze(buyers, categories=["Electronics"])

        # With no candidates, there are no whitespace opportunities reported
        # (this is expected behavior - whitespace is only reported when lookalikes exist)
        assert "Electronics" not in result.category_whitespaces
        assert result.total_opportunities == 0

    def test_whitespace_analysis_with_predictor_and_lookalikes(self):
        """Test whitespace analysis with CLV predictor and actual lookalikes."""
        # Create buyers with very specific engagement pattern
        buyers = [
            make_profile(
                f"buyer_{i}",
                total_revenue=100.0,
                total_purchases=5,
                total_sessions=20 + i,  # Vary slightly
                category_affinities=[
                    CategoryAffinity(
                        category="Electronics",
                        view_count=30 + i,
                        purchase_count=3,
                        engagement_score=0.9,
                    )
                ],
            )
            for i in range(10)
        ]

        # Create candidates with VERY similar engagement patterns
        candidates = [
            make_profile(
                f"candidate_{i}",
                total_revenue=50.0,
                total_purchases=1,
                total_sessions=20 + i,  # Same pattern as buyers
                category_affinities=[
                    CategoryAffinity(
                        category="Electronics",
                        view_count=30 + i,  # Same view pattern
                        purchase_count=0,
                        engagement_score=0.85,
                    )
                ],
            )
            for i in range(5)
        ]

        all_profiles = buyers + candidates

        # Mock predictor
        predictions = {f"buyer_{i}": 500.0 for i in range(10)}
        predictions.update({f"candidate_{i}": 200.0 for i in range(5)})
        mock_predictor = make_mock_predictor(predictions)

        analyzer = WhitespaceAnalyzer(
            min_seed_size=5,
            similarity_threshold=0.0,  # Very low threshold to catch lookalikes
            clv_predictor=mock_predictor,
        )

        result = analyzer.analyze(all_profiles, categories=["Electronics"])

        assert "Electronics" in result.category_whitespaces
        whitespace = result.category_whitespaces["Electronics"]

        # Buyer CLV should use predictions
        assert whitespace.buyer_avg_clv == Decimal("500")
