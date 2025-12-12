"""
Module: predictor

Purpose: CLV prediction using trained ML models.

Provides the CLVPredictor class for making Customer Lifetime Value predictions
from CustomerProfile objects using pre-trained models.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.data.schemas import CustomerProfile
from src.features.clv.features import CLVFeatureBuilder, CLVFeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class CLVPrediction:
    """Result of a CLV prediction for a single customer."""

    customer_id: str
    predicted_clv: Decimal
    confidence: float  # 0.0 to 1.0
    prediction_timestamp: datetime
    model_version: str
    top_features: list[tuple[str, float]]  # (feature_name, importance)

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction to dictionary."""
        return {
            "customer_id": self.customer_id,
            "predicted_clv": float(self.predicted_clv),
            "confidence": self.confidence,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "model_version": self.model_version,
            "top_features": [
                {"feature": name, "importance": importance}
                for name, importance in self.top_features
            ],
        }


@dataclass
class ModelMetadata:
    """Metadata about a trained CLV model."""

    version: str
    trained_at: datetime
    model_type: str  # e.g., "GradientBoostingRegressor"
    feature_names: list[str]
    feature_importances: dict[str, float]
    metrics: dict[str, float]  # e.g., {"mae": 123.45, "rmse": 200.0, "r2": 0.75}
    config: CLVFeatureConfig


class CLVPredictor:
    """Make CLV predictions using trained ML models.

    Loads a pre-trained model and uses CLVFeatureBuilder to transform
    CustomerProfile objects into features for prediction.

    Example:
        >>> predictor = CLVPredictor.load("models/clv_model_v1.joblib")
        >>> prediction = predictor.predict(customer_profile)
        >>> print(f"Predicted CLV: ${prediction.predicted_clv:.2f}")
    """

    def __init__(
        self,
        model: Any,
        metadata: ModelMetadata,
        feature_builder: CLVFeatureBuilder,
    ) -> None:
        """Initialize predictor with trained model.

        Args:
            model: Trained sklearn model with predict() method
            metadata: Model metadata and configuration
            feature_builder: Feature builder configured for this model
        """
        self._model = model
        self._metadata = metadata
        self._feature_builder = feature_builder
        self._feature_names = metadata.feature_names

    @property
    def model_version(self) -> str:
        """Get model version string."""
        return self._metadata.version

    @property
    def feature_names(self) -> list[str]:
        """Get ordered list of feature names."""
        return self._feature_names.copy()

    @property
    def feature_importances(self) -> dict[str, float]:
        """Get feature importance scores."""
        return self._metadata.feature_importances.copy()

    @property
    def metrics(self) -> dict[str, float]:
        """Get training metrics."""
        return self._metadata.metrics.copy()

    @classmethod
    def load(cls, model_path: str | Path) -> "CLVPredictor":
        """Load a trained model from disk.

        Args:
            model_path: Path to saved model file (.joblib)

        Returns:
            Initialized CLVPredictor

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is invalid
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading CLV model from {model_path}")

        try:
            data = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}") from e

        # Validate required fields
        required_fields = ["model", "metadata", "config"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Invalid model file: missing '{field}'")

        # Reconstruct metadata
        meta_dict = data["metadata"]
        metadata = ModelMetadata(
            version=meta_dict["version"],
            trained_at=datetime.fromisoformat(meta_dict["trained_at"]),
            model_type=meta_dict["model_type"],
            feature_names=meta_dict["feature_names"],
            feature_importances=meta_dict["feature_importances"],
            metrics=meta_dict["metrics"],
            config=CLVFeatureConfig(**data["config"]),
        )

        # Create feature builder with same config
        feature_builder = CLVFeatureBuilder(metadata.config)

        return cls(
            model=data["model"],
            metadata=metadata,
            feature_builder=feature_builder,
        )

    def save(self, model_path: str | Path) -> None:
        """Save model to disk.

        Args:
            model_path: Path for saved model file (.joblib)
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize metadata
        meta_dict = {
            "version": self._metadata.version,
            "trained_at": self._metadata.trained_at.isoformat(),
            "model_type": self._metadata.model_type,
            "feature_names": self._metadata.feature_names,
            "feature_importances": self._metadata.feature_importances,
            "metrics": self._metadata.metrics,
        }

        # Serialize config
        config_dict = {
            "include_rfm": self._metadata.config.include_rfm,
            "include_engagement": self._metadata.config.include_engagement,
            "include_intervals": self._metadata.config.include_intervals,
            "include_temporal": self._metadata.config.include_temporal,
            "include_efficiency": self._metadata.config.include_efficiency,
            "include_category": self._metadata.config.include_category,
            "include_behavior": self._metadata.config.include_behavior,
            "include_seasonality": self._metadata.config.include_seasonality,
            "seasonality_period_days": self._metadata.config.seasonality_period_days,
            "max_category_features": self._metadata.config.max_category_features,
            "include_feature_metadata": self._metadata.config.include_feature_metadata,
        }

        data = {
            "model": self._model,
            "metadata": meta_dict,
            "config": config_dict,
        }

        logger.info(f"Saving CLV model to {model_path}")
        joblib.dump(data, model_path)

    def predict(
        self,
        profile: CustomerProfile,
        n_top_features: int = 5,
    ) -> CLVPrediction:
        """Predict CLV for a single customer profile.

        Args:
            profile: CustomerProfile to predict CLV for
            n_top_features: Number of top features to include in result

        Returns:
            CLVPrediction with predicted value and metadata
        """
        # Extract features
        features = self._feature_builder.build_features(profile)

        # Build feature vector in correct order
        feature_vector = np.array([
            [features.get(name, 0.0) for name in self._feature_names]
        ])

        # Make prediction
        predicted_value = self._model.predict(feature_vector)[0]

        # Calculate confidence based on prediction variance if available
        confidence = self._calculate_confidence(feature_vector)

        # Get top contributing features for this prediction
        top_features = self._get_top_features(features, n_top_features)

        return CLVPrediction(
            customer_id=profile.internal_customer_id,
            predicted_clv=Decimal(str(round(max(0, predicted_value), 2))),
            confidence=confidence,
            prediction_timestamp=datetime.now(),
            model_version=self._metadata.version,
            top_features=top_features,
        )

    def predict_batch(
        self,
        profiles: list[CustomerProfile],
        n_top_features: int = 5,
    ) -> list[CLVPrediction]:
        """Predict CLV for multiple customer profiles.

        More efficient than calling predict() repeatedly as it batches
        the feature extraction and prediction.

        Args:
            profiles: List of CustomerProfile objects
            n_top_features: Number of top features to include per prediction

        Returns:
            List of CLVPrediction objects
        """
        if not profiles:
            return []

        # Build feature matrix
        matrix, _ = self._feature_builder.build_feature_matrix(profiles)

        # Ensure feature order matches training
        # (Feature builder maintains consistent order)
        predictions = self._model.predict(matrix)

        # Create CLVPrediction for each
        results = []
        now = datetime.now()

        for i, profile in enumerate(profiles):
            # Get features for this profile to find top contributors
            features = self._feature_builder.build_features(profile)
            top_features = self._get_top_features(features, n_top_features)

            # Get confidence for this prediction
            feature_vector = matrix[i : i + 1]
            confidence = self._calculate_confidence(feature_vector)

            results.append(
                CLVPrediction(
                    customer_id=profile.internal_customer_id,
                    predicted_clv=Decimal(str(round(max(0, predictions[i]), 2))),
                    confidence=confidence,
                    prediction_timestamp=now,
                    model_version=self._metadata.version,
                    top_features=top_features,
                )
            )

        return results

    def _calculate_confidence(self, feature_vector: np.ndarray) -> float:
        """Calculate prediction confidence.

        Uses prediction interval estimation if available (RandomForest only),
        otherwise returns a fixed high confidence.

        Args:
            feature_vector: Feature array for single prediction

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Only use variance-based confidence for RandomForest
        # (GradientBoosting estimators_ is 2D array, not directly usable)
        model_name = type(self._model).__name__

        if model_name == "RandomForestRegressor" and hasattr(self._model, "estimators_"):
            try:
                # Random Forest - use variance across trees
                predictions = np.array([
                    est.predict(feature_vector)[0]
                    for est in self._model.estimators_
                ])
                std = np.std(predictions)
                mean = np.abs(np.mean(predictions))

                # Convert coefficient of variation to confidence
                # Lower CV = higher confidence
                if mean > 0:
                    cv = std / mean
                    confidence = max(0.0, min(1.0, 1.0 - cv))
                else:
                    confidence = 0.5
            except (AttributeError, IndexError):
                confidence = 0.8
        else:
            # For other models (GradientBoosting, Ridge, etc.) use default
            confidence = 0.8

        return round(confidence, 3)

    def _get_top_features(
        self,
        features: dict[str, float],
        n_top: int,
    ) -> list[tuple[str, float]]:
        """Get top contributing features for a prediction.

        Combines feature values with global importance to estimate
        feature contribution.

        Args:
            features: Feature dictionary for this customer
            n_top: Number of top features to return

        Returns:
            List of (feature_name, contribution) tuples
        """
        importances = self._metadata.feature_importances

        # Calculate contribution as value * importance
        contributions = []
        for name, value in features.items():
            if name in importances:
                # Normalize value contribution
                contrib = abs(value) * importances[name]
                contributions.append((name, contrib))

        # Sort by contribution and take top N
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions[:n_top]

    def update_profile_with_prediction(
        self,
        profile: CustomerProfile,
        n_top_features: int = 5,
    ) -> CustomerProfile:
        """Predict CLV and update profile with results.

        Creates a new CustomerProfile instance with CLV prediction fields
        populated.

        Args:
            profile: CustomerProfile to predict and update
            n_top_features: Number of top features to store

        Returns:
            New CustomerProfile with prediction fields set
        """
        prediction = self.predict(profile, n_top_features)

        # Create updated profile with prediction data
        profile_dict = profile.model_dump()
        profile_dict.update({
            "clv_predicted": prediction.predicted_clv,
            "clv_prediction_confidence": prediction.confidence,
            "clv_model_version": prediction.model_version,
            "clv_top_features": [f[0] for f in prediction.top_features],
        })

        return CustomerProfile(**profile_dict)
