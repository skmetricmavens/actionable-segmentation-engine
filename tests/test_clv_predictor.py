"""Tests for CLV predictor module."""

import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.schemas import (
    BehaviorType,
    CategoryAffinity,
    CustomerProfile,
    PurchaseIntervalMetrics,
)
from src.features.clv.features import CLVFeatureConfig
from src.features.clv.predictor import CLVPrediction, CLVPredictor, ModelMetadata


def make_profile(
    internal_customer_id: str = "cust_001",
    total_purchases: int = 10,
    total_revenue: Decimal = Decimal("500.00"),
    total_sessions: int = 20,
    behavior_type: BehaviorType = BehaviorType.REGULAR,
    purchase_intervals: PurchaseIntervalMetrics | None = None,
) -> CustomerProfile:
    """Create a test CustomerProfile."""
    return CustomerProfile(
        internal_customer_id=internal_customer_id,
        first_seen=datetime(2023, 1, 1),
        last_seen=datetime(2024, 1, 1),
        total_purchases=total_purchases,
        total_revenue=total_revenue,
        avg_order_value=total_revenue / total_purchases if total_purchases > 0 else Decimal("0"),
        total_sessions=total_sessions,
        total_page_views=100,
        total_items_viewed=50,
        total_cart_additions=15,
        days_since_last_purchase=30,
        purchase_frequency_per_month=0.83,
        cart_abandonment_rate=0.3,
        total_refunds=Decimal("50.00"),
        refund_rate=0.1,
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
        behavior_type=behavior_type,
        purchase_intervals=purchase_intervals,
    )


class SimpleModel:
    """Simple picklable model for testing."""

    def __init__(self, prediction: float = 1500.0):
        self._prediction = prediction

    def predict(self, X):
        return np.full(len(X), self._prediction)


def create_mock_model():
    """Create a mock sklearn model."""
    model = MagicMock(spec=["predict", "fit"])
    model.predict.return_value = np.array([1500.0])
    return model


def create_picklable_model(prediction: float = 1500.0) -> SimpleModel:
    """Create a picklable model for save/load tests."""
    return SimpleModel(prediction)


def create_mock_ensemble_model():
    """Create a mock ensemble model with estimators."""
    model = MagicMock()
    model.predict.return_value = np.array([1500.0])

    # Create mock estimators for confidence calculation
    estimator1 = MagicMock()
    estimator1.predict.return_value = np.array([1400.0])
    estimator2 = MagicMock()
    estimator2.predict.return_value = np.array([1500.0])
    estimator3 = MagicMock()
    estimator3.predict.return_value = np.array([1600.0])

    model.estimators_ = [estimator1, estimator2, estimator3]
    return model


def create_metadata() -> ModelMetadata:
    """Create test model metadata."""
    return ModelMetadata(
        version="v1.0.0",
        trained_at=datetime(2024, 1, 1),
        model_type="GradientBoostingRegressor",
        feature_names=[
            "days_since_last_purchase",
            "total_purchases",
            "purchase_frequency_per_month",
            "total_revenue",
            "avg_order_value",
        ],
        feature_importances={
            "days_since_last_purchase": 0.3,
            "total_purchases": 0.25,
            "purchase_frequency_per_month": 0.2,
            "total_revenue": 0.15,
            "avg_order_value": 0.1,
        },
        metrics={"mae": 150.0, "rmse": 200.0, "r2": 0.75},
        config=CLVFeatureConfig(
            include_rfm=True,
            include_engagement=False,
            include_intervals=False,
            include_temporal=False,
            include_efficiency=False,
            include_category=False,
            include_behavior=False,
            include_seasonality=False,
        ),
    )


class TestCLVPrediction:
    """Tests for CLVPrediction dataclass."""

    def test_to_dict(self):
        """Test converting prediction to dictionary."""
        prediction = CLVPrediction(
            customer_id="cust_001",
            predicted_clv=Decimal("1500.00"),
            confidence=0.85,
            prediction_timestamp=datetime(2024, 6, 1, 12, 0, 0),
            model_version="v1.0.0",
            top_features=[("total_revenue", 0.4), ("total_purchases", 0.3)],
        )

        result = prediction.to_dict()

        assert result["customer_id"] == "cust_001"
        assert result["predicted_clv"] == 1500.0
        assert result["confidence"] == 0.85
        assert result["model_version"] == "v1.0.0"
        assert len(result["top_features"]) == 2
        assert result["top_features"][0]["feature"] == "total_revenue"


class TestCLVPredictor:
    """Tests for CLVPredictor class."""

    def test_predict_single_profile(self):
        """Test predicting CLV for a single profile."""
        model = create_mock_model()
        metadata = create_metadata()
        from src.features.clv.features import CLVFeatureBuilder

        feature_builder = CLVFeatureBuilder(metadata.config)

        predictor = CLVPredictor(
            model=model,
            metadata=metadata,
            feature_builder=feature_builder,
        )

        profile = make_profile()
        prediction = predictor.predict(profile)

        assert prediction.customer_id == "cust_001"
        assert prediction.predicted_clv == Decimal("1500.00")
        assert prediction.model_version == "v1.0.0"
        assert 0 <= prediction.confidence <= 1.0
        assert len(prediction.top_features) <= 5

    def test_predict_clamps_negative_values(self):
        """Test that negative predictions are clamped to zero."""
        model = MagicMock()
        model.predict.return_value = np.array([-100.0])
        metadata = create_metadata()
        from src.features.clv.features import CLVFeatureBuilder

        feature_builder = CLVFeatureBuilder(metadata.config)

        predictor = CLVPredictor(
            model=model,
            metadata=metadata,
            feature_builder=feature_builder,
        )

        profile = make_profile()
        prediction = predictor.predict(profile)

        assert prediction.predicted_clv == Decimal("0.00")

    def test_predict_batch(self):
        """Test batch prediction for multiple profiles."""
        model = MagicMock()
        model.predict.return_value = np.array([1500.0, 2000.0, 1000.0])
        metadata = create_metadata()
        from src.features.clv.features import CLVFeatureBuilder

        feature_builder = CLVFeatureBuilder(metadata.config)

        predictor = CLVPredictor(
            model=model,
            metadata=metadata,
            feature_builder=feature_builder,
        )

        profiles = [
            make_profile(internal_customer_id="cust_001"),
            make_profile(internal_customer_id="cust_002", total_revenue=Decimal("1000.00")),
            make_profile(internal_customer_id="cust_003", total_revenue=Decimal("200.00")),
        ]

        predictions = predictor.predict_batch(profiles)

        assert len(predictions) == 3
        assert predictions[0].customer_id == "cust_001"
        assert predictions[1].customer_id == "cust_002"
        assert predictions[2].customer_id == "cust_003"

    def test_predict_batch_empty(self):
        """Test batch prediction with empty list."""
        model = create_mock_model()
        metadata = create_metadata()
        from src.features.clv.features import CLVFeatureBuilder

        feature_builder = CLVFeatureBuilder(metadata.config)

        predictor = CLVPredictor(
            model=model,
            metadata=metadata,
            feature_builder=feature_builder,
        )

        predictions = predictor.predict_batch([])
        assert predictions == []

    def test_model_properties(self):
        """Test predictor properties."""
        model = create_mock_model()
        metadata = create_metadata()
        from src.features.clv.features import CLVFeatureBuilder

        feature_builder = CLVFeatureBuilder(metadata.config)

        predictor = CLVPredictor(
            model=model,
            metadata=metadata,
            feature_builder=feature_builder,
        )

        assert predictor.model_version == "v1.0.0"
        assert len(predictor.feature_names) == 5
        assert "total_purchases" in predictor.feature_names
        assert predictor.metrics["mae"] == 150.0
        assert predictor.feature_importances["total_purchases"] == 0.25

    def test_confidence_with_ensemble_model(self):
        """Test confidence calculation with ensemble model."""
        model = create_mock_ensemble_model()
        metadata = create_metadata()
        from src.features.clv.features import CLVFeatureBuilder

        feature_builder = CLVFeatureBuilder(metadata.config)

        predictor = CLVPredictor(
            model=model,
            metadata=metadata,
            feature_builder=feature_builder,
        )

        profile = make_profile()
        prediction = predictor.predict(profile)

        # Ensemble should provide variance-based confidence
        assert 0 < prediction.confidence < 1.0

    def test_update_profile_with_prediction(self):
        """Test updating profile with CLV prediction."""
        model = create_mock_model()
        metadata = create_metadata()
        from src.features.clv.features import CLVFeatureBuilder

        feature_builder = CLVFeatureBuilder(metadata.config)

        predictor = CLVPredictor(
            model=model,
            metadata=metadata,
            feature_builder=feature_builder,
        )

        profile = make_profile()
        updated = predictor.update_profile_with_prediction(profile)

        assert updated.clv_predicted == Decimal("1500.00")
        assert updated.clv_prediction_confidence == 0.8  # Default for non-ensemble
        assert updated.clv_model_version == "v1.0.0"
        assert len(updated.clv_top_features) <= 5


class TestModelPersistence:
    """Tests for model save/load functionality."""

    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        model = create_picklable_model()  # Use picklable model
        metadata = create_metadata()
        from src.features.clv.features import CLVFeatureBuilder

        feature_builder = CLVFeatureBuilder(metadata.config)

        predictor = CLVPredictor(
            model=model,
            metadata=metadata,
            feature_builder=feature_builder,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            predictor.save(model_path)

            assert model_path.exists()

            # Load the model
            loaded = CLVPredictor.load(model_path)

            assert loaded.model_version == "v1.0.0"
            assert loaded.feature_names == predictor.feature_names
            assert loaded.metrics == predictor.metrics

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            CLVPredictor.load("/nonexistent/path/model.joblib")

    def test_load_invalid_file(self):
        """Test loading from invalid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an invalid file
            invalid_path = Path(tmpdir) / "invalid.joblib"
            invalid_path.write_text("not a valid joblib file")

            with pytest.raises(ValueError):
                CLVPredictor.load(invalid_path)

    def test_save_creates_directories(self):
        """Test that save creates parent directories."""
        model = create_picklable_model()  # Use picklable model
        metadata = create_metadata()
        from src.features.clv.features import CLVFeatureBuilder

        feature_builder = CLVFeatureBuilder(metadata.config)

        predictor = CLVPredictor(
            model=model,
            metadata=metadata,
            feature_builder=feature_builder,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dir" / "model.joblib"
            predictor.save(nested_path)

            assert nested_path.exists()


class TestTopFeatures:
    """Tests for feature importance calculations."""

    def test_top_features_limited(self):
        """Test that top features are limited to n_top."""
        model = create_mock_model()
        metadata = create_metadata()
        from src.features.clv.features import CLVFeatureBuilder

        feature_builder = CLVFeatureBuilder(metadata.config)

        predictor = CLVPredictor(
            model=model,
            metadata=metadata,
            feature_builder=feature_builder,
        )

        profile = make_profile()
        prediction = predictor.predict(profile, n_top_features=2)

        assert len(prediction.top_features) <= 2

    def test_top_features_sorted_by_contribution(self):
        """Test that top features are sorted by contribution."""
        model = create_mock_model()
        metadata = create_metadata()
        from src.features.clv.features import CLVFeatureBuilder

        feature_builder = CLVFeatureBuilder(metadata.config)

        predictor = CLVPredictor(
            model=model,
            metadata=metadata,
            feature_builder=feature_builder,
        )

        profile = make_profile()
        prediction = predictor.predict(profile, n_top_features=5)

        # Check that contributions are in descending order
        contributions = [f[1] for f in prediction.top_features]
        assert contributions == sorted(contributions, reverse=True)
