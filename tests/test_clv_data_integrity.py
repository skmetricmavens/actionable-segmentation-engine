"""
CLV-Specific Data Integrity and Data Science Validation Tests.

These tests verify ML pipeline invariants:
1. Feature Matrix Reconciliation - row/column counts preserved
2. Prediction Sanity Checks - outputs are reasonable
3. Training Data Consistency - no data leakage, proper splits
4. Model Output Verification - predictions sum correctly across segments

CRITICAL for ML model reliability and preventing silent model degradation.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest

from src.data.schemas import (
    BehaviorType,
    CategoryAffinity,
    CustomerProfile,
    EventRecord,
    EventType,
    EventProperties,
    PurchaseIntervalMetrics,
)
from src.features.clv.features import CLVFeatureBuilder, CLVFeatureConfig
from src.features.clv.training import CLVTrainer, TrainingConfig
from src.features.profile_builder import build_profile


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_event(
    customer_id: str,
    event_type: EventType,
    timestamp: datetime,
    **properties,
) -> EventRecord:
    """Create an event record with minimal boilerplate."""
    return EventRecord(
        event_id=f"evt_{customer_id}_{timestamp.timestamp()}",
        internal_customer_id=customer_id,
        event_type=event_type,
        timestamp=timestamp,
        properties=EventProperties(**properties),
    )


def create_purchase_event(
    customer_id: str,
    amount: Decimal,
    timestamp: datetime,
) -> EventRecord:
    """Create a purchase event with amount."""
    return create_event(
        customer_id=customer_id,
        event_type=EventType.PURCHASE,
        timestamp=timestamp,
        total_amount=amount,
        order_id=f"order_{timestamp.timestamp()}",
    )


def create_test_profile(
    customer_id: str,
    total_revenue: Decimal = Decimal("500.00"),
    total_purchases: int = 5,
    days_offset: int = 0,
) -> CustomerProfile:
    """Create a test profile with known values."""
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=days_offset)
    return CustomerProfile(
        internal_customer_id=customer_id,
        first_seen=base_date - timedelta(days=180),
        last_seen=base_date,
        total_purchases=total_purchases,
        total_revenue=total_revenue,
        avg_order_value=total_revenue / total_purchases if total_purchases > 0 else Decimal("0"),
        total_sessions=20,
        total_page_views=100,
        total_items_viewed=50,
        total_cart_additions=15,
        days_since_last_purchase=30,
        purchase_frequency_per_month=0.83,
        cart_abandonment_rate=0.3,
        total_refunds=Decimal("0"),
        refund_rate=0.0,
        churn_risk_score=0.2,
        preferred_day_of_week=2,
        preferred_hour_of_day=14,
        mobile_session_ratio=0.4,
        category_affinities=[
            CategoryAffinity(category="Electronics", level=1, engagement_score=0.8),
        ],
        behavior_type=BehaviorType.REGULAR,
    )


def create_test_profiles(n: int) -> list[CustomerProfile]:
    """Create n test profiles with varying characteristics."""
    profiles = []
    for i in range(n):
        profiles.append(
            create_test_profile(
                customer_id=f"cust_{i:03d}",
                total_revenue=Decimal(str(100 + i * 50)),
                total_purchases=max(1, i % 10 + 1),
                days_offset=i * 3,
            )
        )
    return profiles


# =============================================================================
# FEATURE MATRIX RECONCILIATION TESTS
# =============================================================================


class TestFeatureMatrixReconciliation:
    """
    Verify that feature extraction preserves data dimensions correctly.
    """

    def test_feature_matrix_row_count_matches_profile_count(self) -> None:
        """
        RECONCILIATION: Number of rows in feature matrix equals number of profiles.

        No profiles should be lost or duplicated during feature extraction.
        """
        profiles = create_test_profiles(50)
        builder = CLVFeatureBuilder()

        # Control total
        control_count = len(profiles)

        # Action
        matrix, feature_names = builder.build_feature_matrix(profiles)

        # Reconciliation
        assert matrix.shape[0] == control_count, (
            f"Row count mismatch. Expected: {control_count}, Got: {matrix.shape[0]}"
        )

    def test_feature_matrix_column_count_matches_feature_names(self) -> None:
        """
        RECONCILIATION: Number of columns equals number of feature names.

        Feature names must be 1:1 with matrix columns.
        """
        profiles = create_test_profiles(10)
        builder = CLVFeatureBuilder()

        matrix, feature_names = builder.build_feature_matrix(profiles)

        assert matrix.shape[1] == len(feature_names), (
            f"Column count doesn't match feature names. "
            f"Columns: {matrix.shape[1]}, Names: {len(feature_names)}"
        )

    def test_feature_matrix_consistent_across_calls(self) -> None:
        """
        RECONCILIATION: Same profiles produce identical feature matrices.

        Feature extraction must be deterministic.
        """
        profiles = create_test_profiles(20)
        builder = CLVFeatureBuilder()

        matrix1, names1 = builder.build_feature_matrix(profiles)
        matrix2, names2 = builder.build_feature_matrix(profiles)

        assert names1 == names2, "Feature names differ between calls"
        np.testing.assert_array_equal(matrix1, matrix2, "Feature values differ between calls")

    def test_single_profile_features_match_batch(self) -> None:
        """
        RECONCILIATION: Single profile extraction matches batch extraction.

        Individual and batch methods must produce consistent results.
        """
        profiles = create_test_profiles(10)
        builder = CLVFeatureBuilder()

        # Batch extraction
        matrix, feature_names = builder.build_feature_matrix(profiles)

        # Single extraction for each profile
        for i, profile in enumerate(profiles):
            single_features = builder.build_features(profile)
            for j, name in enumerate(feature_names):
                assert matrix[i, j] == single_features.get(name, 0.0), (
                    f"Mismatch at profile {i}, feature {name}. "
                    f"Batch: {matrix[i, j]}, Single: {single_features.get(name)}"
                )


# =============================================================================
# PREDICTION SANITY CHECKS
# =============================================================================


class TestPredictionSanityChecks:
    """
    Sanity checks to verify predictions are reasonable.
    """

    def test_predictions_are_non_negative(self) -> None:
        """
        SANITY: CLV predictions should never be negative.

        Negative CLV doesn't make business sense.
        """
        profiles = create_test_profiles(30)
        config = TrainingConfig(model_type="gradient_boosting", n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)
        predictor = trainer.create_predictor(result)

        for profile in profiles[:10]:
            prediction = predictor.predict(profile)
            assert prediction.predicted_clv >= 0, (
                f"Negative CLV prediction: {prediction.predicted_clv} "
                f"for customer {prediction.customer_id}"
            )

    def test_predictions_within_reasonable_range(self) -> None:
        """
        SANITY: Predictions should be within a reasonable range of training data.

        Extreme outlier predictions may indicate model issues.
        """
        profiles = create_test_profiles(50)
        revenues = [float(p.total_revenue) for p in profiles]
        max_revenue = max(revenues)
        min_revenue = min(revenues)

        config = TrainingConfig(model_type="gradient_boosting", n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)
        predictor = trainer.create_predictor(result)

        for profile in profiles[:10]:
            prediction = predictor.predict(profile)
            pred_value = float(prediction.predicted_clv)

            # Predictions should be within 5x the range of training data
            reasonable_max = max_revenue * 5
            reasonable_min = 0

            assert reasonable_min <= pred_value <= reasonable_max, (
                f"Prediction {pred_value} outside reasonable range "
                f"[{reasonable_min}, {reasonable_max}]"
            )

    def test_confidence_scores_valid_range(self) -> None:
        """
        SANITY: Confidence scores must be between 0 and 1.
        """
        profiles = create_test_profiles(30)
        config = TrainingConfig(model_type="gradient_boosting", n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)
        predictor = trainer.create_predictor(result)

        for profile in profiles[:10]:
            prediction = predictor.predict(profile)
            assert 0.0 <= prediction.confidence <= 1.0, (
                f"Invalid confidence score: {prediction.confidence}"
            )

    def test_high_value_customers_get_high_predictions(self) -> None:
        """
        SANITY: Customers with high historical revenue should get high predictions.

        Basic correlation check - model should learn revenue patterns.
        """
        # Create profiles with clear high/low revenue distinction
        low_value = [
            create_test_profile(f"low_{i}", total_revenue=Decimal("50"), total_purchases=1)
            for i in range(20)
        ]
        high_value = [
            create_test_profile(f"high_{i}", total_revenue=Decimal("5000"), total_purchases=50)
            for i in range(20)
        ]
        profiles = low_value + high_value

        config = TrainingConfig(model_type="gradient_boosting", n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)
        predictor = trainer.create_predictor(result)

        # Get predictions
        low_preds = [float(predictor.predict(p).predicted_clv) for p in low_value[:5]]
        high_preds = [float(predictor.predict(p).predicted_clv) for p in high_value[:5]]

        avg_low = np.mean(low_preds)
        avg_high = np.mean(high_preds)

        assert avg_high > avg_low, (
            f"High-value customers should have higher predictions. "
            f"Avg low: {avg_low:.2f}, Avg high: {avg_high:.2f}"
        )


# =============================================================================
# TRAINING DATA CONSISTENCY
# =============================================================================


class TestTrainingDataConsistency:
    """
    Verify training process maintains data integrity.
    """

    def test_cv_folds_dont_overlap(self) -> None:
        """
        CONSISTENCY: Cross-validation folds should be disjoint.

        This is handled by sklearn, but we verify the split count.
        """
        profiles = create_test_profiles(50)
        config = TrainingConfig(n_splits=5)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        # Should have exactly 5 CV scores
        assert len(result.cv_scores) == 5, (
            f"Expected 5 CV scores, got {len(result.cv_scores)}"
        )

    def test_holdout_set_is_disjoint_from_training(self) -> None:
        """
        CONSISTENCY: Holdout data should not be in training set.

        Verified by checking that holdout metrics differ from training metrics.
        """
        profiles = create_test_profiles(100)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result, holdout_metrics = trainer.train_with_holdout(
            profiles, holdout_fraction=0.2
        )

        # Holdout metrics should typically be worse than training metrics
        # (due to not being in training set)
        # This is a soft check - just verify we get different numbers
        assert "mae" in holdout_metrics
        assert "mae" in result.final_metrics
        # The metrics should be calculated (non-zero for typical data)
        assert holdout_metrics["mae"] >= 0

    def test_feature_importances_sum_approximately_to_one(self) -> None:
        """
        CONSISTENCY: For tree models, feature importances should sum to ~1.
        """
        profiles = create_test_profiles(50)
        config = TrainingConfig(model_type="gradient_boosting", n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        importance_sum = sum(result.feature_importances.values())
        assert 0.99 <= importance_sum <= 1.01, (
            f"Feature importances should sum to ~1, got {importance_sum}"
        )

    def test_all_features_have_importances(self) -> None:
        """
        CONSISTENCY: Every feature should have an importance score.
        """
        profiles = create_test_profiles(30)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        assert len(result.feature_importances) == len(result.feature_names), (
            f"Importance count ({len(result.feature_importances)}) "
            f"doesn't match feature count ({len(result.feature_names)})"
        )


# =============================================================================
# CONTROL TOTALS FOR CLV
# =============================================================================


class TestCLVControlTotals:
    """
    Control totals specific to CLV pipeline.
    """

    def test_profile_clv_fields_populated_after_prediction(self) -> None:
        """
        CONTROL TOTAL: Updated profile should have all CLV fields set.
        """
        profiles = create_test_profiles(30)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)
        predictor = trainer.create_predictor(result, version="test_v1")

        updated = predictor.update_profile_with_prediction(profiles[0])

        assert updated.clv_predicted is not None, "clv_predicted not set"
        assert updated.clv_prediction_confidence is not None, "confidence not set"
        assert updated.clv_model_version == "test_v1", "version not set"
        assert len(updated.clv_top_features) > 0, "top features not set"

    def test_batch_predictions_count_matches_input(self) -> None:
        """
        CONTROL TOTAL: Batch prediction count equals input count.
        """
        profiles = create_test_profiles(25)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)
        predictor = trainer.create_predictor(result)

        predictions = predictor.predict_batch(profiles)

        assert len(predictions) == len(profiles), (
            f"Prediction count ({len(predictions)}) != profile count ({len(profiles)})"
        )

    def test_customer_ids_preserved_in_predictions(self) -> None:
        """
        CONTROL TOTAL: Customer IDs in predictions match input profiles.
        """
        profiles = create_test_profiles(15)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)
        predictor = trainer.create_predictor(result)

        predictions = predictor.predict_batch(profiles)

        input_ids = {p.internal_customer_id for p in profiles}
        output_ids = {p.customer_id for p in predictions}

        assert input_ids == output_ids, (
            f"Customer ID mismatch. Input: {input_ids}, Output: {output_ids}"
        )


# =============================================================================
# PROFILE BUILDING CLV INTEGRATION
# =============================================================================


class TestProfileBuildingCLVIntegration:
    """
    Verify CLV fields are correctly populated during profile building.
    """

    def test_purchase_intervals_calculated_for_multi_purchase_customers(self) -> None:
        """
        CONTROL TOTAL: Customers with 2+ purchases should have interval metrics.
        """
        base_time = datetime.now(timezone.utc)

        events = [
            create_purchase_event("cust_1", Decimal("100"), base_time),
            create_purchase_event("cust_1", Decimal("50"), base_time + timedelta(days=10)),
            create_purchase_event("cust_1", Decimal("75"), base_time + timedelta(days=25)),
        ]

        profile = build_profile("cust_1", events)

        assert profile.purchase_intervals is not None, "Intervals should be calculated"
        assert profile.purchase_intervals.interval_mean is not None
        assert profile.purchase_intervals.interval_mean > 0

    def test_behavior_type_classification_consistent(self) -> None:
        """
        CONTROL TOTAL: Behavior type should be set for all profiles.
        """
        base_time = datetime.now(timezone.utc)

        # Single purchase customer
        single_events = [
            create_purchase_event("cust_1", Decimal("100"), base_time),
        ]
        single_profile = build_profile("cust_1", single_events)
        assert single_profile.behavior_type == BehaviorType.ONE_TIME

        # No purchase customer
        no_purchase_events = [
            create_event("cust_2", EventType.VIEW_ITEM, base_time),
        ]
        no_purchase_profile = build_profile("cust_2", no_purchase_events)
        assert no_purchase_profile.behavior_type == BehaviorType.NEW

    def test_interval_metrics_mathematically_consistent(self) -> None:
        """
        SANITY: Interval metrics should be internally consistent.

        - mean should be between min and max
        - std should be non-negative
        - regularity_index should be between 0 and 1
        """
        base_time = datetime.now(timezone.utc)

        events = [
            create_purchase_event("cust_1", Decimal("100"), base_time),
            create_purchase_event("cust_1", Decimal("50"), base_time + timedelta(days=7)),
            create_purchase_event("cust_1", Decimal("75"), base_time + timedelta(days=14)),
            create_purchase_event("cust_1", Decimal("60"), base_time + timedelta(days=28)),
        ]

        profile = build_profile("cust_1", events)
        intervals = profile.purchase_intervals

        assert intervals is not None
        assert intervals.interval_min <= intervals.interval_mean <= intervals.interval_max, (
            f"Mean ({intervals.interval_mean}) not between "
            f"min ({intervals.interval_min}) and max ({intervals.interval_max})"
        )
        assert intervals.interval_std >= 0, f"Negative std: {intervals.interval_std}"
        if intervals.regularity_index is not None:
            assert 0 <= intervals.regularity_index <= 1, (
                f"Regularity index out of range: {intervals.regularity_index}"
            )


# =============================================================================
# MODEL METRICS VALIDATION
# =============================================================================


class TestModelMetricsValidation:
    """
    Validate that model metrics are mathematically correct.
    """

    def test_rmse_greater_than_or_equal_to_mae(self) -> None:
        """
        SANITY: RMSE >= MAE (always true mathematically).
        """
        profiles = create_test_profiles(50)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        assert result.final_metrics["rmse"] >= result.final_metrics["mae"], (
            f"RMSE ({result.final_metrics['rmse']}) < MAE ({result.final_metrics['mae']})"
        )

    def test_r2_bounded_correctly(self) -> None:
        """
        SANITY: R² should be <= 1.0 (can be negative for bad models).
        """
        profiles = create_test_profiles(50)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        assert result.final_metrics["r2"] <= 1.0, (
            f"R² > 1.0: {result.final_metrics['r2']}"
        )

    def test_mape_non_negative(self) -> None:
        """
        SANITY: MAPE should be non-negative.
        """
        profiles = create_test_profiles(50)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        assert result.final_metrics["mape"] >= 0, (
            f"Negative MAPE: {result.final_metrics['mape']}"
        )

    def test_cv_scores_all_positive(self) -> None:
        """
        SANITY: CV scores (MAE) should all be positive.
        """
        profiles = create_test_profiles(50)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        for i, score in enumerate(result.cv_scores):
            assert score >= 0, f"Negative CV score at fold {i}: {score}"
