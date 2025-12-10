"""
Module: sensitivity

Purpose: Sensitivity analysis for segment robustness testing.

Tests segment stability under various perturbations:
- Feature dropping (leave-one-out stability)
- Time window variations
- Sampling stability (bootstrap)

Architecture Notes:
- Pure functional core with stateful analyzer class
- Uses clustering from clusterer module
- Produces RobustnessScore for each segment
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.data.schemas import (
    CustomerProfile,
    FeatureSensitivityResult,
    RobustnessScore,
    Segment,
    TimeWindowSensitivityResult,
)
from src.exceptions import (
    FeatureSensitivityError,
    InsufficientDataError,
    SensitivityTestError,
    TimeWindowSensitivityError,
)
from src.segmentation.clusterer import (
    ClusteringResult,
    CustomerClusterer,
    FeatureMatrix,
    cluster_customers,
    create_segments_from_clusters,
    extract_features,
    standardize_features,
)


@dataclass
class FeatureDropResult:
    """Result from dropping a single feature."""

    dropped_feature: str
    labels: NDArray[np.int64]
    silhouette: float | None
    inertia: float
    cluster_sizes: dict[int, int]


@dataclass
class StabilityMetrics:
    """Metrics for comparing cluster assignments."""

    adjusted_rand_index: float
    normalized_mutual_info: float
    membership_jaccard: float


def compute_adjusted_rand_index(
    labels1: NDArray[np.int64],
    labels2: NDArray[np.int64],
) -> float:
    """
    Compute Adjusted Rand Index between two clusterings.

    ARI measures similarity between cluster assignments, adjusted for chance.
    Range: -1 to 1, with 1 being perfect agreement.

    Args:
        labels1: First cluster assignment
        labels2: Second cluster assignment

    Returns:
        Adjusted Rand Index score
    """
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(labels1, labels2))


def compute_normalized_mutual_info(
    labels1: NDArray[np.int64],
    labels2: NDArray[np.int64],
) -> float:
    """
    Compute Normalized Mutual Information between clusterings.

    NMI measures shared information, normalized to [0, 1].

    Args:
        labels1: First cluster assignment
        labels2: Second cluster assignment

    Returns:
        NMI score
    """
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(labels1, labels2))


def compute_membership_jaccard(
    labels1: NDArray[np.int64],
    labels2: NDArray[np.int64],
) -> float:
    """
    Compute average Jaccard similarity of cluster memberships.

    For each sample, computes Jaccard of co-members across clusterings.

    Args:
        labels1: First cluster assignment
        labels2: Second cluster assignment

    Returns:
        Average Jaccard similarity
    """
    n_samples = len(labels1)
    if n_samples < 2:
        return 1.0

    # Build co-membership matrices
    comembership1 = labels1[:, None] == labels1[None, :]
    comembership2 = labels2[:, None] == labels2[None, :]

    # Compute Jaccard: |intersection| / |union|
    intersection = np.sum(comembership1 & comembership2)
    union = np.sum(comembership1 | comembership2)

    if union == 0:
        return 1.0

    return float(intersection / union)


def compute_stability_metrics(
    labels1: NDArray[np.int64],
    labels2: NDArray[np.int64],
) -> StabilityMetrics:
    """
    Compute all stability metrics between two clusterings.

    Args:
        labels1: First cluster assignment
        labels2: Second cluster assignment

    Returns:
        StabilityMetrics with ARI, NMI, and Jaccard scores
    """
    return StabilityMetrics(
        adjusted_rand_index=compute_adjusted_rand_index(labels1, labels2),
        normalized_mutual_info=compute_normalized_mutual_info(labels1, labels2),
        membership_jaccard=compute_membership_jaccard(labels1, labels2),
    )


def drop_feature_from_matrix(
    feature_matrix: FeatureMatrix,
    feature_to_drop: str,
) -> FeatureMatrix:
    """
    Create new FeatureMatrix with one feature removed.

    Args:
        feature_matrix: Original feature matrix
        feature_to_drop: Name of feature to remove

    Returns:
        New FeatureMatrix without the specified feature

    Raises:
        ValueError: If feature not found
    """
    if feature_to_drop not in feature_matrix.feature_names:
        raise ValueError(f"Feature '{feature_to_drop}' not in matrix")

    idx = feature_matrix.feature_names.index(feature_to_drop)
    new_features = [f for f in feature_matrix.feature_names if f != feature_to_drop]
    new_matrix = np.delete(feature_matrix.matrix, idx, axis=1)

    return FeatureMatrix(
        matrix=new_matrix,
        feature_names=new_features,
        customer_ids=feature_matrix.customer_ids,
        scaler=None,  # Scaler no longer valid after dropping
    )


def run_feature_sensitivity_test(
    profiles: list[CustomerProfile],
    *,
    n_clusters: int = 5,
    random_seed: int = 42,
    stability_threshold: float = 0.7,
) -> tuple[list[FeatureDropResult], list[str]]:
    """
    Run clustering stability test when individual features are dropped.

    Performs leave-one-out analysis for each feature.

    Args:
        profiles: Customer profiles
        n_clusters: Number of clusters
        random_seed: Random seed for reproducibility
        stability_threshold: ARI threshold for considering stable (default 0.7)

    Returns:
        Tuple of (drop results, list of critical features)

    Raises:
        InsufficientDataError: If insufficient profiles
        FeatureSensitivityError: If test fails critically
    """
    if len(profiles) < n_clusters + 1:
        raise InsufficientDataError(
            f"Need at least {n_clusters + 1} profiles for sensitivity testing",
            required=n_clusters + 1,
            actual=len(profiles),
            data_type="profiles",
        )

    # Get baseline clustering
    features = extract_features(profiles)
    standardized = standardize_features(features)

    try:
        baseline_result = cluster_customers(
            standardized,
            n_clusters=n_clusters,
            random_seed=random_seed,
        )
    except Exception as e:
        raise FeatureSensitivityError(
            f"Baseline clustering failed: {e}",
            dropped_features=[],
        ) from e

    baseline_labels = baseline_result.labels
    drop_results: list[FeatureDropResult] = []
    critical_features: list[str] = []

    # Test each feature
    for feature_name in features.feature_names:
        try:
            # Drop feature and re-cluster
            reduced = drop_feature_from_matrix(features, feature_name)

            # Need at least 1 feature remaining
            if reduced.matrix.shape[1] < 1:
                continue

            reduced_std = standardize_features(reduced)
            result = cluster_customers(
                reduced_std,
                n_clusters=n_clusters,
                random_seed=random_seed,
            )

            # Compare to baseline
            ari = compute_adjusted_rand_index(baseline_labels, result.labels)

            drop_results.append(
                FeatureDropResult(
                    dropped_feature=feature_name,
                    labels=result.labels,
                    silhouette=result.silhouette,
                    inertia=result.inertia,
                    cluster_sizes=result.cluster_sizes,
                )
            )

            # Check if this is a critical feature
            if ari < stability_threshold:
                critical_features.append(feature_name)

        except Exception:
            # Feature drop failed - consider it critical
            critical_features.append(feature_name)

    return drop_results, critical_features


def calculate_feature_stability_score(
    profiles: list[CustomerProfile],
    *,
    n_clusters: int = 5,
    random_seed: int = 42,
) -> FeatureSensitivityResult:
    """
    Calculate overall feature stability score.

    Args:
        profiles: Customer profiles
        n_clusters: Number of clusters
        random_seed: Random seed

    Returns:
        FeatureSensitivityResult with stability metrics
    """
    segment_id = "aggregate"  # For overall analysis

    try:
        drop_results, critical_features = run_feature_sensitivity_test(
            profiles,
            n_clusters=n_clusters,
            random_seed=random_seed,
        )

        n_features = len(drop_results) + len(
            [f for f in critical_features if f not in [r.dropped_feature for r in drop_results]]
        )
        n_stable = len(drop_results) - len(critical_features)

        # Stability is proportion of features that don't significantly change clustering
        stability = n_stable / n_features if n_features > 0 else 0.0

        return FeatureSensitivityResult(
            segment_id=segment_id,
            feature_stability=stability,
            critical_features=critical_features,
            iterations_passed=n_stable,
            total_iterations=n_features,
        )

    except (InsufficientDataError, FeatureSensitivityError) as e:
        return FeatureSensitivityResult(
            segment_id=segment_id,
            feature_stability=0.0,
            critical_features=[],
            iterations_passed=0,
            total_iterations=0,
        )


def filter_profiles_by_time_window(
    profiles: list[CustomerProfile],
    events_cutoff_date: datetime,
) -> list[CustomerProfile]:
    """
    Filter profiles to only include those with activity before cutoff.

    This simulates using a shorter time window for analysis.

    Args:
        profiles: Full list of profiles
        events_cutoff_date: Only include profiles with last_seen before this date

    Returns:
        Filtered list of profiles
    """
    return [p for p in profiles if p.last_seen <= events_cutoff_date]


def run_time_window_sensitivity_test(
    profiles: list[CustomerProfile],
    *,
    window_days_list: list[int] | None = None,
    n_clusters: int = 5,
    random_seed: int = 42,
    reference_date: datetime | None = None,
) -> dict[int, ClusteringResult]:
    """
    Run clustering stability test across different time windows.

    Args:
        profiles: Full customer profiles
        window_days_list: List of window sizes in days (default: [30, 60, 90])
        n_clusters: Number of clusters
        random_seed: Random seed
        reference_date: Reference date for window calculation

    Returns:
        Dictionary mapping window_days to ClusteringResult

    Raises:
        TimeWindowSensitivityError: If test fails
    """
    if window_days_list is None:
        window_days_list = [30, 60, 90]

    if not profiles:
        raise TimeWindowSensitivityError(
            "No profiles provided",
            windows_tested=[],
        )

    # Determine reference date from profiles
    if reference_date is None:
        reference_date = max(p.last_seen for p in profiles)

    results: dict[int, ClusteringResult] = {}

    for window_days in window_days_list:
        cutoff = reference_date - timedelta(days=window_days)

        # Filter profiles that were active during this window
        window_profiles = [
            p for p in profiles
            if p.first_seen <= reference_date and p.last_seen >= cutoff
        ]

        if len(window_profiles) < n_clusters + 1:
            # Not enough profiles for this window
            continue

        try:
            features = extract_features(window_profiles)
            standardized = standardize_features(features)
            result = cluster_customers(
                standardized,
                n_clusters=min(n_clusters, len(window_profiles) - 1),
                random_seed=random_seed,
            )
            results[window_days] = result
        except Exception:
            # Skip this window if clustering fails
            continue

    if not results:
        raise TimeWindowSensitivityError(
            "No valid time windows could be processed",
            windows_tested=window_days_list,
        )

    return results


def calculate_time_window_consistency(
    window_results: dict[int, ClusteringResult],
    profile_id_mapping: dict[int, list[str]],
) -> float:
    """
    Calculate consistency score across time windows.

    Uses pairwise ARI comparison between windows.

    Args:
        window_results: Results from run_time_window_sensitivity_test
        profile_id_mapping: Mapping of window_days to customer_ids for alignment

    Returns:
        Consistency score (0-1)
    """
    if len(window_results) < 2:
        return 1.0  # Single window is trivially consistent

    windows = sorted(window_results.keys())
    pairwise_scores: list[float] = []

    for i, w1 in enumerate(windows[:-1]):
        for w2 in windows[i + 1:]:
            result1 = window_results[w1]
            result2 = window_results[w2]

            # Find common customers
            ids1 = set(profile_id_mapping.get(w1, []))
            ids2 = set(profile_id_mapping.get(w2, []))
            common_ids = ids1 & ids2

            if len(common_ids) < 2:
                continue

            # Get labels for common customers only
            id_to_idx1 = {id_: i for i, id_ in enumerate(profile_id_mapping.get(w1, []))}
            id_to_idx2 = {id_: i for i, id_ in enumerate(profile_id_mapping.get(w2, []))}

            common_labels1 = [result1.labels[id_to_idx1[id_]] for id_ in common_ids if id_ in id_to_idx1]
            common_labels2 = [result2.labels[id_to_idx2[id_]] for id_ in common_ids if id_ in id_to_idx2]

            if len(common_labels1) >= 2:
                ari = compute_adjusted_rand_index(
                    np.array(common_labels1),
                    np.array(common_labels2),
                )
                pairwise_scores.append(ari)

    if not pairwise_scores:
        return 1.0

    return float(np.mean(pairwise_scores))


def calculate_time_window_stability(
    profiles: list[CustomerProfile],
    *,
    window_days_list: list[int] | None = None,
    n_clusters: int = 5,
    random_seed: int = 42,
) -> TimeWindowSensitivityResult:
    """
    Calculate time window sensitivity result.

    Args:
        profiles: Customer profiles
        window_days_list: Windows to test
        n_clusters: Number of clusters
        random_seed: Random seed

    Returns:
        TimeWindowSensitivityResult
    """
    if window_days_list is None:
        window_days_list = [30, 60, 90]

    segment_id = "aggregate"

    try:
        reference_date = max(p.last_seen for p in profiles)
        window_results: dict[int, ClusteringResult] = {}
        profile_id_mapping: dict[int, list[str]] = {}

        for window_days in window_days_list:
            cutoff = reference_date - timedelta(days=window_days)
            window_profiles = [
                p for p in profiles
                if p.first_seen <= reference_date and p.last_seen >= cutoff
            ]

            if len(window_profiles) < n_clusters + 1:
                continue

            features = extract_features(window_profiles)
            standardized = standardize_features(features)
            result = cluster_customers(
                standardized,
                n_clusters=min(n_clusters, len(window_profiles) - 1),
                random_seed=random_seed,
            )
            window_results[window_days] = result
            profile_id_mapping[window_days] = features.customer_ids

        if len(window_results) < 2:
            return TimeWindowSensitivityResult(
                segment_id=segment_id,
                time_consistency=1.0,  # Can't compare with < 2 windows
                windows_tested=list(window_results.keys()),
                stable_across_windows=True,
            )

        consistency = calculate_time_window_consistency(window_results, profile_id_mapping)

        return TimeWindowSensitivityResult(
            segment_id=segment_id,
            time_consistency=consistency,
            windows_tested=list(window_results.keys()),
            stable_across_windows=consistency >= 0.7,
        )

    except Exception:
        return TimeWindowSensitivityResult(
            segment_id=segment_id,
            time_consistency=0.0,
            windows_tested=[],
            stable_across_windows=False,
        )


def bootstrap_sample(
    profiles: list[CustomerProfile],
    *,
    sample_size: int | None = None,
    random_state: np.random.Generator | None = None,
) -> list[CustomerProfile]:
    """
    Create bootstrap sample of profiles.

    Args:
        profiles: Original profiles
        sample_size: Size of bootstrap sample (default: same as original)
        random_state: NumPy random generator

    Returns:
        Bootstrap sample of profiles
    """
    if random_state is None:
        random_state = np.random.default_rng()

    n = len(profiles)
    if sample_size is None:
        sample_size = n

    indices = random_state.choice(n, size=sample_size, replace=True)
    return [profiles[i] for i in indices]


def run_sampling_stability_test(
    profiles: list[CustomerProfile],
    *,
    n_bootstrap: int = 10,
    n_clusters: int = 5,
    random_seed: int = 42,
) -> float:
    """
    Run clustering stability test under bootstrap sampling.

    Args:
        profiles: Customer profiles
        n_bootstrap: Number of bootstrap iterations
        n_clusters: Number of clusters
        random_seed: Base random seed

    Returns:
        Average stability score across bootstrap samples
    """
    if len(profiles) < n_clusters + 1:
        return 0.0

    # Baseline clustering
    features = extract_features(profiles)
    standardized = standardize_features(features)

    try:
        baseline = cluster_customers(
            standardized,
            n_clusters=n_clusters,
            random_seed=random_seed,
        )
    except Exception:
        return 0.0

    # Map customer_ids to indices for comparison
    id_to_idx = {cid: i for i, cid in enumerate(features.customer_ids)}

    stability_scores: list[float] = []
    rng = np.random.default_rng(random_seed)

    for _ in range(n_bootstrap):
        # Bootstrap sample
        sample = bootstrap_sample(profiles, random_state=rng)

        if len(sample) < n_clusters + 1:
            continue

        try:
            sample_features = extract_features(sample)
            sample_std = standardize_features(sample_features)
            sample_result = cluster_customers(
                sample_std,
                n_clusters=n_clusters,
                random_seed=random_seed,
            )

            # Find common customers for comparison
            common_ids = set(sample_features.customer_ids) & set(features.customer_ids)
            if len(common_ids) < 2:
                continue

            # Extract labels for common customers
            sample_id_to_idx = {cid: i for i, cid in enumerate(sample_features.customer_ids)}

            baseline_labels = []
            sample_labels = []
            for cid in common_ids:
                if cid in id_to_idx and cid in sample_id_to_idx:
                    baseline_labels.append(baseline.labels[id_to_idx[cid]])
                    sample_labels.append(sample_result.labels[sample_id_to_idx[cid]])

            if len(baseline_labels) >= 2:
                ari = compute_adjusted_rand_index(
                    np.array(baseline_labels),
                    np.array(sample_labels),
                )
                stability_scores.append(ari)

        except Exception:
            continue

    if not stability_scores:
        return 0.0

    return float(np.mean(stability_scores))


@dataclass
class SensitivityAnalysisResult:
    """Complete sensitivity analysis result."""

    feature_sensitivity: FeatureSensitivityResult
    time_window_sensitivity: TimeWindowSensitivityResult
    sampling_stability: float | None
    robustness_scores: dict[str, RobustnessScore] = field(default_factory=dict)

    @property
    def overall_robustness(self) -> float:
        """Calculate overall robustness from components."""
        base = (
            0.6 * self.feature_sensitivity.feature_stability
            + 0.4 * self.time_window_sensitivity.time_consistency
        )

        if self.sampling_stability is not None:
            base = (
                0.5 * self.feature_sensitivity.feature_stability
                + 0.3 * self.time_window_sensitivity.time_consistency
                + 0.2 * self.sampling_stability
            )

        return base


class SensitivityAnalyzer:
    """
    High-level sensitivity analysis for segment robustness.

    Coordinates feature, time window, and sampling stability tests.
    """

    def __init__(
        self,
        *,
        n_clusters: int = 5,
        random_seed: int = 42,
        n_bootstrap: int = 10,
        window_days_list: list[int] | None = None,
        stability_threshold: float = 0.7,
    ) -> None:
        """
        Initialize SensitivityAnalyzer.

        Args:
            n_clusters: Number of clusters for analysis
            random_seed: Random seed for reproducibility
            n_bootstrap: Number of bootstrap iterations
            window_days_list: Time windows to test
            stability_threshold: Threshold for considering stable
        """
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.n_bootstrap = n_bootstrap
        self.window_days_list = window_days_list or [30, 60, 90]
        self.stability_threshold = stability_threshold

        self._last_result: SensitivityAnalysisResult | None = None

    def analyze(
        self,
        profiles: list[CustomerProfile],
        *,
        include_sampling: bool = True,
    ) -> SensitivityAnalysisResult:
        """
        Run complete sensitivity analysis.

        Args:
            profiles: Customer profiles to analyze
            include_sampling: Whether to include bootstrap sampling test

        Returns:
            SensitivityAnalysisResult with all metrics
        """
        # Feature sensitivity
        feature_result = calculate_feature_stability_score(
            profiles,
            n_clusters=self.n_clusters,
            random_seed=self.random_seed,
        )

        # Time window sensitivity
        time_result = calculate_time_window_stability(
            profiles,
            window_days_list=self.window_days_list,
            n_clusters=self.n_clusters,
            random_seed=self.random_seed,
        )

        # Sampling stability (optional, more expensive)
        sampling_stability = None
        if include_sampling:
            sampling_stability = run_sampling_stability_test(
                profiles,
                n_bootstrap=self.n_bootstrap,
                n_clusters=self.n_clusters,
                random_seed=self.random_seed,
            )

        # Build robustness score
        robustness = RobustnessScore.calculate(
            segment_id="aggregate",
            feature_stability=feature_result.feature_stability,
            time_window_consistency=time_result.time_consistency,
            sampling_stability=sampling_stability,
        )

        result = SensitivityAnalysisResult(
            feature_sensitivity=feature_result,
            time_window_sensitivity=time_result,
            sampling_stability=sampling_stability,
            robustness_scores={"aggregate": robustness},
        )

        self._last_result = result
        return result

    def analyze_segments(
        self,
        profiles: list[CustomerProfile],
        segments: list[Segment],
        *,
        include_sampling: bool = False,
    ) -> dict[str, RobustnessScore]:
        """
        Analyze robustness for each segment individually.

        Args:
            profiles: All customer profiles
            segments: Segments to analyze
            include_sampling: Include bootstrap sampling (expensive)

        Returns:
            Dictionary mapping segment_id to RobustnessScore
        """
        # Build profile lookup
        profile_lookup = {p.internal_customer_id: p for p in profiles}

        results: dict[str, RobustnessScore] = {}

        for segment in segments:
            # Get profiles in this segment
            segment_profile_ids = {m.internal_customer_id for m in segment.members}
            segment_profiles = [
                profile_lookup[pid]
                for pid in segment_profile_ids
                if pid in profile_lookup
            ]

            if len(segment_profiles) < 3:
                # Too small for meaningful analysis
                results[segment.segment_id] = RobustnessScore.calculate(
                    segment_id=segment.segment_id,
                    feature_stability=0.0,
                    time_window_consistency=0.0,
                )
                continue

            # Run analysis on segment profiles
            # Note: We use a simplified version for per-segment analysis
            # since we can't cluster within a single segment

            # For per-segment, we use the overall analysis weighted by segment characteristics
            # This is a simplified approach - full per-segment would need different methodology

            # Feature sensitivity: check if segment characteristics are stable
            segment_feature_stability = self._estimate_segment_feature_stability(
                segment_profiles,
                segment,
            )

            # Time window: check if segment membership is stable over time
            segment_time_stability = self._estimate_segment_time_stability(
                segment_profiles,
                segment,
            )

            # Sampling stability (if requested)
            segment_sampling = None
            if include_sampling:
                segment_sampling = self._estimate_segment_sampling_stability(
                    segment_profiles,
                    segment,
                )

            results[segment.segment_id] = RobustnessScore.calculate(
                segment_id=segment.segment_id,
                feature_stability=segment_feature_stability,
                time_window_consistency=segment_time_stability,
                sampling_stability=segment_sampling,
            )

        return results

    def _estimate_segment_feature_stability(
        self,
        segment_profiles: list[CustomerProfile],
        segment: Segment,
    ) -> float:
        """Estimate feature stability for a segment based on characteristic variance."""
        if len(segment_profiles) < 2:
            return 0.0

        # Calculate coefficient of variation for key features
        revenues = [float(p.total_revenue) for p in segment_profiles]
        clvs = [float(p.clv_estimate) for p in segment_profiles]
        sessions = [float(p.total_sessions) for p in segment_profiles]

        def cv(values: list[float]) -> float:
            if not values or np.mean(values) == 0:
                return 1.0
            return float(np.std(values) / (np.mean(values) + 1e-10))

        # Lower CV = more stable segment characteristics
        avg_cv = (cv(revenues) + cv(clvs) + cv(sessions)) / 3

        # Convert to stability score (lower CV = higher stability)
        stability = max(0.0, 1.0 - avg_cv)
        return min(1.0, stability)

    def _estimate_segment_time_stability(
        self,
        segment_profiles: list[CustomerProfile],
        segment: Segment,
    ) -> float:
        """Estimate time stability based on profile tenure distribution."""
        if len(segment_profiles) < 2:
            return 0.0

        # Check tenure distribution - stable segments have consistent tenure
        tenures = [(p.last_seen - p.first_seen).days for p in segment_profiles]

        if not tenures or max(tenures) == 0:
            return 0.5

        # Consistency based on tenure CV
        tenure_cv = float(np.std(tenures) / (np.mean(tenures) + 1e-10))
        stability = max(0.0, 1.0 - tenure_cv * 0.5)
        return min(1.0, stability)

    def _estimate_segment_sampling_stability(
        self,
        segment_profiles: list[CustomerProfile],
        segment: Segment,
    ) -> float:
        """Estimate sampling stability based on segment size and variance."""
        if len(segment_profiles) < 5:
            return 0.3  # Small segments less stable under sampling

        # Larger segments with lower variance are more sampling-stable
        size_factor = min(1.0, len(segment_profiles) / 50)

        revenues = [float(p.total_revenue) for p in segment_profiles]
        revenue_cv = float(np.std(revenues) / (np.mean(revenues) + 1e-10)) if revenues else 1.0
        variance_factor = max(0.0, 1.0 - revenue_cv * 0.5)

        return size_factor * 0.5 + variance_factor * 0.5

    @property
    def last_result(self) -> SensitivityAnalysisResult | None:
        """Get the last analysis result."""
        return self._last_result


def get_sensitivity_summary(result: SensitivityAnalysisResult) -> dict[str, Any]:
    """
    Generate human-readable summary of sensitivity analysis.

    Args:
        result: SensitivityAnalysisResult

    Returns:
        Dictionary with summary information
    """
    summary: dict[str, Any] = {
        "overall_robustness": result.overall_robustness,
        "feature_stability": {
            "score": result.feature_sensitivity.feature_stability,
            "critical_features": result.feature_sensitivity.critical_features,
            "iterations": f"{result.feature_sensitivity.iterations_passed}/{result.feature_sensitivity.total_iterations}",
        },
        "time_window_stability": {
            "score": result.time_window_sensitivity.time_consistency,
            "windows_tested": result.time_window_sensitivity.windows_tested,
            "stable": result.time_window_sensitivity.stable_across_windows,
        },
    }

    if result.sampling_stability is not None:
        summary["sampling_stability"] = {
            "score": result.sampling_stability,
        }

    # Add tier classification
    if result.overall_robustness >= 0.75:
        summary["tier"] = "HIGH"
        summary["recommendation"] = "Segment is production-ready"
    elif result.overall_robustness >= 0.5:
        summary["tier"] = "MEDIUM"
        summary["recommendation"] = "Segment requires monitoring"
    else:
        summary["tier"] = "LOW"
        summary["recommendation"] = "Segment may be unstable, consider refinement"

    return summary
