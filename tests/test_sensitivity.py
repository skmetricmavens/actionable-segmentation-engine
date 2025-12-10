"""
Tests for sensitivity analysis module.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest

from src.data.schemas import (
    CustomerProfile,
    FeatureSensitivityResult,
    RobustnessScore,
    RobustnessTier,
    Segment,
    SegmentMember,
    TimeWindowSensitivityResult,
)
from src.exceptions import InsufficientDataError
from src.segmentation.clusterer import (
    FeatureMatrix,
    extract_features,
)
from src.segmentation.sensitivity import (
    SensitivityAnalysisResult,
    SensitivityAnalyzer,
    bootstrap_sample,
    calculate_feature_stability_score,
    calculate_time_window_consistency,
    calculate_time_window_stability,
    compute_adjusted_rand_index,
    compute_membership_jaccard,
    compute_normalized_mutual_info,
    compute_stability_metrics,
    drop_feature_from_matrix,
    get_sensitivity_summary,
    run_feature_sensitivity_test,
    run_sampling_stability_test,
    run_time_window_sensitivity_test,
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
    days_offset: int = 0,
) -> CustomerProfile:
    """Helper to create CustomerProfile for testing."""
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return CustomerProfile(
        internal_customer_id=customer_id,
        first_seen=base_date + timedelta(days=days_offset),
        last_seen=base_date + timedelta(days=90 + days_offset),
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
    """Create diverse profiles for testing."""
    profiles = []

    # High-value customers
    for i in range(n // 4):
        profiles.append(
            make_profile(
                f"high_value_{i}",
                total_revenue=500.0 + i * 50,
                clv_estimate=1000.0 + i * 100,
                total_purchases=10 + i,
                total_sessions=15 + i,
                mobile_session_ratio=0.2,
                days_offset=i,
            )
        )

    # Low-value browsers
    for i in range(n // 4):
        profiles.append(
            make_profile(
                f"browser_{i}",
                total_revenue=20.0 + i * 5,
                clv_estimate=50.0 + i * 10,
                total_purchases=1,
                total_sessions=20 + i,
                mobile_session_ratio=0.8,
                days_offset=i + 10,
            )
        )

    # Moderate customers
    for i in range(n // 4):
        profiles.append(
            make_profile(
                f"moderate_{i}",
                total_revenue=150.0 + i * 20,
                clv_estimate=300.0 + i * 30,
                total_purchases=5 + i,
                total_sessions=8 + i,
                mobile_session_ratio=0.5,
                days_offset=i + 20,
            )
        )

    # Churning customers
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
                days_offset=i + 30,
            )
        )

    return profiles


def make_segment(
    segment_id: str,
    profile_ids: list[str],
) -> Segment:
    """Create a segment from profile IDs."""
    return Segment(
        segment_id=segment_id,
        name=f"Test Segment {segment_id}",
        description="Test segment",
        members=[SegmentMember(internal_customer_id=pid) for pid in profile_ids],
        size=len(profile_ids),
        total_clv=Decimal("1000"),
        avg_clv=Decimal(str(1000 / max(len(profile_ids), 1))),
    )


# =============================================================================
# STABILITY METRICS TESTS
# =============================================================================


class TestStabilityMetrics:
    """Tests for stability metric computation."""

    def test_ari_identical_labels(self) -> None:
        """Test ARI with identical labelings."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        ari = compute_adjusted_rand_index(labels, labels)
        assert ari == pytest.approx(1.0)

    def test_ari_different_labels(self) -> None:
        """Test ARI with different labelings."""
        labels1 = np.array([0, 0, 0, 1, 1, 1])
        labels2 = np.array([0, 1, 2, 0, 1, 2])
        ari = compute_adjusted_rand_index(labels1, labels2)
        assert -1 <= ari <= 1

    def test_nmi_identical_labels(self) -> None:
        """Test NMI with identical labelings."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        nmi = compute_normalized_mutual_info(labels, labels)
        assert nmi == pytest.approx(1.0)

    def test_nmi_different_labels(self) -> None:
        """Test NMI range."""
        labels1 = np.array([0, 0, 1, 1])
        labels2 = np.array([0, 1, 0, 1])
        nmi = compute_normalized_mutual_info(labels1, labels2)
        assert 0 <= nmi <= 1

    def test_jaccard_identical_labels(self) -> None:
        """Test Jaccard with identical labelings."""
        labels = np.array([0, 0, 1, 1])
        jaccard = compute_membership_jaccard(labels, labels)
        assert jaccard == pytest.approx(1.0)

    def test_jaccard_different_labels(self) -> None:
        """Test Jaccard range."""
        labels1 = np.array([0, 0, 1, 1])
        labels2 = np.array([0, 1, 0, 1])
        jaccard = compute_membership_jaccard(labels1, labels2)
        assert 0 <= jaccard <= 1

    def test_compute_all_metrics(self) -> None:
        """Test combined stability metrics."""
        labels1 = np.array([0, 0, 1, 1, 2, 2])
        labels2 = np.array([0, 0, 1, 1, 2, 2])
        metrics = compute_stability_metrics(labels1, labels2)

        assert metrics.adjusted_rand_index == pytest.approx(1.0)
        assert metrics.normalized_mutual_info == pytest.approx(1.0)
        assert metrics.membership_jaccard == pytest.approx(1.0)


# =============================================================================
# FEATURE DROPPING TESTS
# =============================================================================


class TestFeatureDropping:
    """Tests for feature dropping functionality."""

    def test_drop_feature_from_matrix(self) -> None:
        """Test dropping a single feature."""
        profiles = make_diverse_profiles(10)
        features = extract_features(profiles)

        original_n_features = len(features.feature_names)
        dropped = drop_feature_from_matrix(features, "total_revenue")

        assert len(dropped.feature_names) == original_n_features - 1
        assert "total_revenue" not in dropped.feature_names
        assert dropped.matrix.shape[1] == original_n_features - 1

    def test_drop_nonexistent_feature(self) -> None:
        """Test error when dropping nonexistent feature."""
        profiles = make_diverse_profiles(10)
        features = extract_features(profiles)

        with pytest.raises(ValueError, match="not in matrix"):
            drop_feature_from_matrix(features, "nonexistent_feature")

    def test_drop_preserves_customer_ids(self) -> None:
        """Test that dropping preserves customer IDs."""
        profiles = make_diverse_profiles(10)
        features = extract_features(profiles)

        dropped = drop_feature_from_matrix(features, "total_sessions")
        assert dropped.customer_ids == features.customer_ids


# =============================================================================
# FEATURE SENSITIVITY TESTS
# =============================================================================


class TestFeatureSensitivity:
    """Tests for feature sensitivity analysis."""

    def test_feature_sensitivity_basic(self) -> None:
        """Test basic feature sensitivity analysis."""
        profiles = make_diverse_profiles(20)

        drop_results, critical_features = run_feature_sensitivity_test(
            profiles,
            n_clusters=4,
            random_seed=42,
        )

        assert len(drop_results) > 0
        # Each result should have the expected fields
        for result in drop_results:
            assert result.dropped_feature in [
                "total_revenue", "avg_order_value", "clv_estimate",
                "total_purchases", "purchase_frequency", "churn_risk",
                "total_sessions", "total_page_views", "total_items_viewed",
                "cart_abandonment_rate", "preferred_day", "preferred_hour",
                "mobile_ratio",
            ]

    def test_feature_sensitivity_insufficient_data(self) -> None:
        """Test error with insufficient data."""
        profiles = [make_profile(f"cust_{i}") for i in range(3)]

        with pytest.raises(InsufficientDataError):
            run_feature_sensitivity_test(profiles, n_clusters=5)

    def test_calculate_feature_stability_score(self) -> None:
        """Test overall feature stability score calculation."""
        profiles = make_diverse_profiles(20)

        result = calculate_feature_stability_score(
            profiles,
            n_clusters=4,
            random_seed=42,
        )

        assert isinstance(result, FeatureSensitivityResult)
        assert 0 <= result.feature_stability <= 1
        assert result.total_iterations > 0


# =============================================================================
# TIME WINDOW SENSITIVITY TESTS
# =============================================================================


class TestTimeWindowSensitivity:
    """Tests for time window sensitivity analysis."""

    def test_time_window_sensitivity_basic(self) -> None:
        """Test basic time window sensitivity."""
        profiles = make_diverse_profiles(40)

        results = run_time_window_sensitivity_test(
            profiles,
            window_days_list=[30, 60, 90],
            n_clusters=4,
            random_seed=42,
        )

        # Should get results for at least some windows
        assert len(results) > 0

    def test_calculate_time_window_stability(self) -> None:
        """Test time window stability calculation."""
        profiles = make_diverse_profiles(40)

        result = calculate_time_window_stability(
            profiles,
            window_days_list=[30, 60, 90],
            n_clusters=4,
            random_seed=42,
        )

        assert isinstance(result, TimeWindowSensitivityResult)
        assert 0 <= result.time_consistency <= 1

    def test_time_window_consistency_calculation(self) -> None:
        """Test consistency calculation with mock data."""
        from src.segmentation.clusterer import cluster_customers, standardize_features

        profiles = make_diverse_profiles(20)
        features = extract_features(profiles)
        std_features = standardize_features(features)

        result1 = cluster_customers(std_features, n_clusters=4, random_seed=42)
        result2 = cluster_customers(std_features, n_clusters=4, random_seed=42)

        # Same clustering should be perfectly consistent
        window_results = {30: result1, 60: result2}
        profile_id_mapping = {
            30: features.customer_ids,
            60: features.customer_ids,
        }

        consistency = calculate_time_window_consistency(window_results, profile_id_mapping)
        assert consistency == pytest.approx(1.0)


# =============================================================================
# SAMPLING STABILITY TESTS
# =============================================================================


class TestSamplingStability:
    """Tests for sampling stability analysis."""

    def test_bootstrap_sample(self) -> None:
        """Test bootstrap sampling."""
        profiles = make_diverse_profiles(20)

        rng = np.random.default_rng(42)
        sample = bootstrap_sample(profiles, random_state=rng)

        assert len(sample) == len(profiles)
        # Bootstrap allows duplicates, so unique count may differ
        unique_ids = {p.internal_customer_id for p in sample}
        assert len(unique_ids) <= len(profiles)

    def test_bootstrap_sample_custom_size(self) -> None:
        """Test bootstrap with custom sample size."""
        profiles = make_diverse_profiles(20)

        rng = np.random.default_rng(42)
        sample = bootstrap_sample(profiles, sample_size=10, random_state=rng)

        assert len(sample) == 10

    def test_sampling_stability_basic(self) -> None:
        """Test sampling stability calculation."""
        profiles = make_diverse_profiles(30)

        stability = run_sampling_stability_test(
            profiles,
            n_bootstrap=5,
            n_clusters=4,
            random_seed=42,
        )

        assert 0 <= stability <= 1

    def test_sampling_stability_insufficient_data(self) -> None:
        """Test sampling stability with insufficient data."""
        profiles = [make_profile(f"cust_{i}") for i in range(3)]

        stability = run_sampling_stability_test(
            profiles,
            n_bootstrap=5,
            n_clusters=5,
        )

        assert stability == 0.0


# =============================================================================
# SENSITIVITY ANALYZER CLASS TESTS
# =============================================================================


class TestSensitivityAnalyzer:
    """Tests for SensitivityAnalyzer class."""

    def test_analyzer_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = SensitivityAnalyzer(
            n_clusters=4,
            random_seed=42,
            n_bootstrap=5,
        )

        assert analyzer.n_clusters == 4
        assert analyzer.random_seed == 42
        assert analyzer.n_bootstrap == 5

    def test_analyze_basic(self) -> None:
        """Test basic analysis."""
        profiles = make_diverse_profiles(30)
        analyzer = SensitivityAnalyzer(n_clusters=4, n_bootstrap=3)

        result = analyzer.analyze(profiles, include_sampling=False)

        assert isinstance(result, SensitivityAnalysisResult)
        assert result.feature_sensitivity is not None
        assert result.time_window_sensitivity is not None
        assert analyzer.last_result is not None

    def test_analyze_with_sampling(self) -> None:
        """Test analysis including sampling stability."""
        profiles = make_diverse_profiles(30)
        analyzer = SensitivityAnalyzer(n_clusters=4, n_bootstrap=3)

        result = analyzer.analyze(profiles, include_sampling=True)

        assert result.sampling_stability is not None

    def test_analyze_overall_robustness(self) -> None:
        """Test overall robustness calculation."""
        profiles = make_diverse_profiles(30)
        analyzer = SensitivityAnalyzer(n_clusters=4, n_bootstrap=3)

        result = analyzer.analyze(profiles)

        assert 0 <= result.overall_robustness <= 1

    def test_analyze_segments(self) -> None:
        """Test per-segment analysis."""
        profiles = make_diverse_profiles(40)

        # Create segments from profile IDs
        segments = [
            make_segment("seg_0", [p.internal_customer_id for p in profiles[:10]]),
            make_segment("seg_1", [p.internal_customer_id for p in profiles[10:20]]),
            make_segment("seg_2", [p.internal_customer_id for p in profiles[20:30]]),
            make_segment("seg_3", [p.internal_customer_id for p in profiles[30:]]),
        ]

        analyzer = SensitivityAnalyzer(n_clusters=4)
        results = analyzer.analyze_segments(profiles, segments)

        assert len(results) == 4
        for seg_id, score in results.items():
            assert isinstance(score, RobustnessScore)
            assert seg_id in ["seg_0", "seg_1", "seg_2", "seg_3"]

    def test_analyze_small_segment(self) -> None:
        """Test analysis of very small segments."""
        profiles = make_diverse_profiles(20)

        # Create one tiny segment
        segments = [
            make_segment("tiny", [profiles[0].internal_customer_id]),
        ]

        analyzer = SensitivityAnalyzer(n_clusters=4)
        results = analyzer.analyze_segments(profiles, segments)

        # Should still return a score (low stability)
        assert "tiny" in results
        assert results["tiny"].feature_stability == 0.0


# =============================================================================
# ROBUSTNESS SCORE TESTS
# =============================================================================


class TestRobustnessScore:
    """Tests for RobustnessScore calculation."""

    def test_robustness_score_high(self) -> None:
        """Test high robustness tier."""
        score = RobustnessScore.calculate(
            segment_id="test",
            feature_stability=0.9,
            time_window_consistency=0.8,
        )

        assert score.robustness_tier == RobustnessTier.HIGH
        assert score.is_production_ready is True
        assert score.overall_robustness >= 0.75

    def test_robustness_score_medium(self) -> None:
        """Test medium robustness tier."""
        score = RobustnessScore.calculate(
            segment_id="test",
            feature_stability=0.6,
            time_window_consistency=0.6,
        )

        assert score.robustness_tier == RobustnessTier.MEDIUM
        assert score.requires_monitoring is True

    def test_robustness_score_low(self) -> None:
        """Test low robustness tier."""
        score = RobustnessScore.calculate(
            segment_id="test",
            feature_stability=0.3,
            time_window_consistency=0.2,
        )

        assert score.robustness_tier == RobustnessTier.LOW
        assert score.is_production_ready is False

    def test_robustness_with_optional_scores(self) -> None:
        """Test robustness calculation with optional scores."""
        score = RobustnessScore.calculate(
            segment_id="test",
            feature_stability=0.8,
            time_window_consistency=0.7,
            sampling_stability=0.9,
            threshold_robustness=0.85,
        )

        # Should use adjusted weights
        assert score.sampling_stability == 0.9
        assert score.threshold_robustness == 0.85


# =============================================================================
# SUMMARY TESTS
# =============================================================================


class TestSensitivitySummary:
    """Tests for sensitivity summary generation."""

    def test_get_sensitivity_summary(self) -> None:
        """Test summary generation."""
        profiles = make_diverse_profiles(30)
        analyzer = SensitivityAnalyzer(n_clusters=4, n_bootstrap=3)
        result = analyzer.analyze(profiles)

        summary = get_sensitivity_summary(result)

        assert "overall_robustness" in summary
        assert "feature_stability" in summary
        assert "time_window_stability" in summary
        assert "tier" in summary
        assert "recommendation" in summary

    def test_summary_tiers(self) -> None:
        """Test summary tier classification."""
        # Create mock results for different tiers
        high_result = SensitivityAnalysisResult(
            feature_sensitivity=FeatureSensitivityResult(
                segment_id="test",
                feature_stability=0.9,
                critical_features=[],
                iterations_passed=10,
                total_iterations=10,
            ),
            time_window_sensitivity=TimeWindowSensitivityResult(
                segment_id="test",
                time_consistency=0.8,
                windows_tested=[30, 60, 90],
                stable_across_windows=True,
            ),
            sampling_stability=None,
        )

        summary = get_sensitivity_summary(high_result)
        assert summary["tier"] == "HIGH"

        low_result = SensitivityAnalysisResult(
            feature_sensitivity=FeatureSensitivityResult(
                segment_id="test",
                feature_stability=0.2,
                critical_features=["total_revenue", "clv_estimate"],
                iterations_passed=2,
                total_iterations=10,
            ),
            time_window_sensitivity=TimeWindowSensitivityResult(
                segment_id="test",
                time_consistency=0.3,
                windows_tested=[30, 60],
                stable_across_windows=False,
            ),
            sampling_stability=None,
        )

        summary = get_sensitivity_summary(low_result)
        assert summary["tier"] == "LOW"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestSensitivityIntegration:
    """Integration tests with synthetic data."""

    def test_end_to_end_sensitivity_analysis(self) -> None:
        """Test complete sensitivity analysis pipeline."""
        from src.data.joiner import resolve_customer_merges
        from src.data.synthetic_generator import SyntheticDataGenerator
        from src.features.profile_builder import build_profiles_batch
        from src.segmentation.clusterer import CustomerClusterer

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

        # Sensitivity analysis
        analyzer = SensitivityAnalyzer(n_clusters=4, n_bootstrap=3)
        result = analyzer.analyze(profiles, include_sampling=True)

        # Verify results
        assert result.feature_sensitivity is not None
        assert result.time_window_sensitivity is not None
        assert result.sampling_stability is not None
        assert 0 <= result.overall_robustness <= 1

        # Per-segment analysis
        segment_scores = analyzer.analyze_segments(profiles, segments)
        assert len(segment_scores) == len(segments)

        # Summary
        summary = get_sensitivity_summary(result)
        assert summary["tier"] in ["HIGH", "MEDIUM", "LOW"]
