"""Tests for CLV training module."""

import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from src.data.schemas import BehaviorType, CategoryAffinity, CustomerProfile
from src.features.clv.training import (
    CLVTrainer,
    TrainingConfig,
    train_clv_model,
)


def make_profile(
    internal_customer_id: str,
    total_revenue: Decimal,
    last_seen: datetime,
    total_purchases: int = 5,
) -> CustomerProfile:
    """Create a test CustomerProfile."""
    return CustomerProfile(
        internal_customer_id=internal_customer_id,
        first_seen=last_seen - timedelta(days=180),
        last_seen=last_seen,
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
        total_refunds=Decimal("0.00"),
        refund_rate=0.0,
        churn_risk_score=0.2,
        preferred_day_of_week=2,
        preferred_hour_of_day=14,
        mobile_session_ratio=0.4,
        category_affinities=[
            CategoryAffinity(
                category="Electronics",
                level=1,
                engagement_score=0.8,
                purchase_count=5,
                view_count=20,
            ),
        ],
        behavior_type=BehaviorType.REGULAR,
    )


def create_training_profiles(n: int = 50) -> list[CustomerProfile]:
    """Create a list of profiles with varying revenue and dates."""
    profiles = []
    base_date = datetime(2024, 1, 1)

    for i in range(n):
        # Create profiles with correlated features and revenue
        revenue = Decimal(str(100 + i * 50 + np.random.randint(-20, 20)))
        purchases = max(1, 5 + i // 10)
        last_seen = base_date + timedelta(days=i * 3)

        profiles.append(
            make_profile(
                internal_customer_id=f"cust_{i:03d}",
                total_revenue=revenue,
                last_seen=last_seen,
                total_purchases=purchases,
            )
        )

    return profiles


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.model_type == "gradient_boosting"
        assert config.n_splits == 5
        assert config.target_column == "total_revenue"

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            model_type="random_forest",
            n_splits=3,
            model_params={"n_estimators": 50},
        )

        assert config.model_type == "random_forest"
        assert config.n_splits == 3
        assert config.model_params["n_estimators"] == 50


class TestCLVTrainer:
    """Tests for CLVTrainer."""

    def test_train_gradient_boosting(self):
        """Test training with gradient boosting model."""
        profiles = create_training_profiles(30)
        config = TrainingConfig(model_type="gradient_boosting", n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        assert result.model is not None
        assert len(result.feature_names) > 0
        assert len(result.feature_importances) == len(result.feature_names)
        assert len(result.cv_scores) == 3
        assert "mae" in result.final_metrics
        assert "rmse" in result.final_metrics
        assert "r2" in result.final_metrics

    def test_train_random_forest(self):
        """Test training with random forest model."""
        profiles = create_training_profiles(30)
        config = TrainingConfig(model_type="random_forest", n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        assert result.model is not None
        assert result.model_type == "random_forest"

    def test_train_ridge_regression(self):
        """Test training with ridge regression model."""
        profiles = create_training_profiles(30)
        config = TrainingConfig(model_type="ridge", n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        assert result.model is not None
        assert result.model_type == "ridge"

    def test_train_with_custom_targets(self):
        """Test training with explicit target values."""
        profiles = create_training_profiles(30)
        custom_targets = [float(i * 100) for i in range(30)]

        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles, targets=custom_targets)

        assert result.model is not None

    def test_train_with_holdout(self):
        """Test training with holdout evaluation."""
        profiles = create_training_profiles(50)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result, holdout_metrics = trainer.train_with_holdout(
            profiles, holdout_fraction=0.2
        )

        assert result.model is not None
        assert "mae" in holdout_metrics
        assert "rmse" in holdout_metrics
        assert "r2" in holdout_metrics

    def test_create_predictor(self):
        """Test creating predictor from training result."""
        profiles = create_training_profiles(30)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)
        predictor = trainer.create_predictor(result, version="v1.0.0")

        assert predictor.model_version == "v1.0.0"
        assert predictor.feature_names == result.feature_names

        # Test prediction works
        prediction = predictor.predict(profiles[0])
        assert prediction.predicted_clv >= 0

    def test_save_model(self):
        """Test saving trained model."""
        profiles = create_training_profiles(30)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            saved_path = trainer.save_model(result, model_path)

            assert saved_path.exists()

    def test_unsupported_model_type(self):
        """Test error handling for unsupported model type."""
        config = TrainingConfig(model_type="unsupported")  # type: ignore
        trainer = CLVTrainer(config)
        profiles = create_training_profiles(10)

        with pytest.raises(ValueError, match="Unsupported model type"):
            trainer.train(profiles)

    def test_feature_importances_sum_to_one(self):
        """Test that feature importances are normalized for linear models."""
        profiles = create_training_profiles(30)
        config = TrainingConfig(model_type="ridge", n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)
        importance_sum = sum(result.feature_importances.values())

        # Should be approximately 1.0 for normalized importances
        assert 0.99 < importance_sum < 1.01

    def test_time_aware_sorting(self):
        """Test that profiles are sorted by time for training."""
        # Create profiles with intentionally unordered dates
        profiles = []
        base_date = datetime(2024, 1, 1)

        for i in [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]:  # Shuffled order
            profiles.append(
                make_profile(
                    internal_customer_id=f"cust_{i}",
                    total_revenue=Decimal(str(100 * (i + 1))),
                    last_seen=base_date + timedelta(days=i * 10),
                )
            )

        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        # Should not raise - time-aware CV handles sorting
        result = trainer.train(profiles)
        assert result.model is not None


class TestConvenienceFunction:
    """Tests for train_clv_model convenience function."""

    def test_train_clv_model_basic(self):
        """Test basic usage of convenience function."""
        profiles = create_training_profiles(30)

        predictor = train_clv_model(profiles)

        assert predictor is not None
        prediction = predictor.predict(profiles[0])
        assert prediction.predicted_clv >= 0

    def test_train_clv_model_with_save(self):
        """Test saving model via convenience function."""
        profiles = create_training_profiles(30)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            predictor = train_clv_model(profiles, save_path=model_path)

            assert predictor is not None
            assert model_path.exists()


class TestMetrics:
    """Tests for metric calculations."""

    def test_mape_calculation(self):
        """Test MAPE calculation handles edge cases."""
        profiles = create_training_profiles(30)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        assert "mape" in result.final_metrics
        assert result.final_metrics["mape"] >= 0

    def test_metrics_reasonable_range(self):
        """Test that metrics are in reasonable ranges."""
        profiles = create_training_profiles(50)
        config = TrainingConfig(n_splits=3)
        trainer = CLVTrainer(config)

        result = trainer.train(profiles)

        # R2 should be between -inf and 1.0
        assert result.final_metrics["r2"] <= 1.0

        # MAE and RMSE should be non-negative
        assert result.final_metrics["mae"] >= 0
        assert result.final_metrics["rmse"] >= 0

        # RMSE >= MAE (always true)
        assert result.final_metrics["rmse"] >= result.final_metrics["mae"]
