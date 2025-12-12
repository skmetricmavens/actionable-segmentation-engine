"""
Module: training

Purpose: Training pipeline for CLV prediction models.

Provides functionality for training, validating, and persisting CLV prediction
models using time-aware data splitting for proper evaluation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.data.schemas import CustomerProfile
from src.features.clv.features import CLVFeatureBuilder, CLVFeatureConfig
from src.features.clv.predictor import CLVPredictor, ModelMetadata

logger = logging.getLogger(__name__)

ModelType = Literal["gradient_boosting", "random_forest", "ridge"]


@dataclass
class TrainingConfig:
    """Configuration for CLV model training."""

    # Model selection
    model_type: ModelType = "gradient_boosting"

    # Time-series cross-validation
    n_splits: int = 5

    # Target column configuration
    target_column: str = "total_revenue"  # Which profile field to predict

    # Model hyperparameters (model-specific)
    model_params: dict[str, Any] = field(default_factory=dict)

    # Feature configuration
    feature_config: CLVFeatureConfig = field(default_factory=CLVFeatureConfig)

    # Output paths
    model_output_path: str | None = None


@dataclass
class TrainingResult:
    """Results from model training."""

    model: Any
    feature_names: list[str]
    feature_importances: dict[str, float]
    cv_scores: list[float]
    final_metrics: dict[str, float]
    training_time: float
    model_type: str
    config: TrainingConfig


class CLVTrainer:
    """Train CLV prediction models from customer profiles.

    Supports multiple model types with time-aware cross-validation to
    prevent data leakage from future observations.

    Example:
        >>> trainer = CLVTrainer(config)
        >>> result = trainer.train(profiles, targets)
        >>> predictor = trainer.create_predictor(result)
    """

    SUPPORTED_MODELS: dict[ModelType, type] = {
        "gradient_boosting": GradientBoostingRegressor,
        "random_forest": RandomForestRegressor,
        "ridge": Ridge,
    }

    DEFAULT_PARAMS: dict[ModelType, dict[str, Any]] = {
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
        },
        "ridge": {
            "alpha": 1.0,
        },
    }

    def __init__(self, config: TrainingConfig | None = None) -> None:
        """Initialize trainer with configuration.

        Args:
            config: Training configuration. Uses defaults if None.
        """
        self.config = config or TrainingConfig()
        self._feature_builder = CLVFeatureBuilder(self.config.feature_config)

    def train(
        self,
        profiles: list[CustomerProfile],
        targets: list[float] | None = None,
    ) -> TrainingResult:
        """Train a CLV prediction model.

        Args:
            profiles: List of CustomerProfile objects for training
            targets: Optional target values. If None, extracts from profiles
                using config.target_column

        Returns:
            TrainingResult with trained model and metrics
        """
        import time

        start_time = time.time()

        # Extract target values if not provided
        if targets is None:
            targets = self._extract_targets(profiles)

        # Build feature matrix
        logger.info(f"Building feature matrix for {len(profiles)} profiles")
        X, feature_names = self._feature_builder.build_feature_matrix(profiles)
        y = np.array(targets)

        # Sort by last_seen for time-aware splitting
        logger.info("Sorting profiles by last_seen for time-aware splitting")
        sorted_indices = np.argsort([p.last_seen for p in profiles])
        X = X[sorted_indices]
        y = y[sorted_indices]

        # Create model
        model = self._create_model()

        # Time-series cross-validation
        logger.info(f"Running {self.config.n_splits}-fold time-series cross-validation")
        cv_scores = self._time_series_cv(model, X, y)
        logger.info(f"CV scores: {cv_scores}, mean: {np.mean(cv_scores):.4f}")

        # Train final model on all data
        logger.info("Training final model on full dataset")
        model.fit(X, y)

        # Get feature importances
        feature_importances = self._get_feature_importances(model, feature_names)

        # Calculate final metrics
        y_pred = model.predict(X)
        final_metrics = self._calculate_metrics(y, y_pred)

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f}s")

        return TrainingResult(
            model=model,
            feature_names=feature_names,
            feature_importances=feature_importances,
            cv_scores=cv_scores.tolist(),
            final_metrics=final_metrics,
            training_time=training_time,
            model_type=self.config.model_type,
            config=self.config,
        )

    def train_with_holdout(
        self,
        profiles: list[CustomerProfile],
        targets: list[float] | None = None,
        holdout_fraction: float = 0.2,
    ) -> tuple[TrainingResult, dict[str, float]]:
        """Train with a time-based holdout set for final evaluation.

        Args:
            profiles: List of CustomerProfile objects
            targets: Optional target values
            holdout_fraction: Fraction of most recent data to hold out

        Returns:
            Tuple of (TrainingResult, holdout_metrics)
        """
        if targets is None:
            targets = self._extract_targets(profiles)

        # Sort by last_seen
        sorted_data = sorted(
            zip(profiles, targets), key=lambda x: x[0].last_seen
        )
        profiles = [p for p, _ in sorted_data]
        targets = [t for _, t in sorted_data]

        # Split into train and holdout
        split_idx = int(len(profiles) * (1 - holdout_fraction))
        train_profiles = profiles[:split_idx]
        train_targets = targets[:split_idx]
        holdout_profiles = profiles[split_idx:]
        holdout_targets = targets[split_idx:]

        logger.info(
            f"Training on {len(train_profiles)} profiles, "
            f"holding out {len(holdout_profiles)} most recent profiles"
        )

        # Train on training set
        result = self.train(train_profiles, train_targets)

        # Evaluate on holdout
        X_holdout, _ = self._feature_builder.build_feature_matrix(holdout_profiles)
        y_holdout = np.array(holdout_targets)
        y_pred_holdout = result.model.predict(X_holdout)

        holdout_metrics = self._calculate_metrics(y_holdout, y_pred_holdout)
        logger.info(f"Holdout metrics: {holdout_metrics}")

        return result, holdout_metrics

    def create_predictor(
        self,
        result: TrainingResult,
        version: str | None = None,
    ) -> CLVPredictor:
        """Create a CLVPredictor from training results.

        Args:
            result: TrainingResult from training
            version: Model version string. Auto-generated if None.

        Returns:
            Configured CLVPredictor ready for inference
        """
        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        metadata = ModelMetadata(
            version=version,
            trained_at=datetime.now(),
            model_type=result.model_type,
            feature_names=result.feature_names,
            feature_importances=result.feature_importances,
            metrics=result.final_metrics,
            config=result.config.feature_config,
        )

        return CLVPredictor(
            model=result.model,
            metadata=metadata,
            feature_builder=self._feature_builder,
        )

    def save_model(
        self,
        result: TrainingResult,
        path: str | Path | None = None,
        version: str | None = None,
    ) -> Path:
        """Save trained model to disk.

        Args:
            result: TrainingResult to save
            path: Output path. Uses config path if None.
            version: Model version string

        Returns:
            Path to saved model file
        """
        if path is None:
            path = self.config.model_output_path
        if path is None:
            raise ValueError("No output path specified")

        path = Path(path)
        predictor = self.create_predictor(result, version)
        predictor.save(path)

        logger.info(f"Model saved to {path}")
        return path

    def _extract_targets(self, profiles: list[CustomerProfile]) -> list[float]:
        """Extract target values from profiles.

        Args:
            profiles: List of CustomerProfile objects

        Returns:
            List of target values
        """
        target_col = self.config.target_column
        targets = []

        for profile in profiles:
            if hasattr(profile, target_col):
                value = getattr(profile, target_col)
                targets.append(float(value))
            else:
                raise ValueError(
                    f"Profile missing target column: {target_col}"
                )

        return targets

    def _create_model(self) -> Any:
        """Create model instance based on configuration."""
        model_class = self.SUPPORTED_MODELS.get(self.config.model_type)
        if model_class is None:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

        # Get default params and override with config
        params = self.DEFAULT_PARAMS.get(self.config.model_type, {}).copy()
        params.update(self.config.model_params)

        return model_class(**params)

    def _time_series_cv(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Perform time-series cross-validation.

        Uses expanding window approach where each fold uses all prior
        data for training and the next time period for validation.

        Args:
            model: Model instance
            X: Feature matrix
            y: Target values

        Returns:
            Array of cross-validation scores (negative MAE)
        """
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)

        # Use negative MAE for scoring (sklearn convention)
        scores = cross_val_score(
            model, X, y, cv=tscv, scoring="neg_mean_absolute_error"
        )

        # Convert back to positive MAE
        return -scores

    def _get_feature_importances(
        self,
        model: Any,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Extract feature importances from trained model.

        Args:
            model: Trained model
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            # For linear models, use absolute coefficient values
            importances = np.abs(model.coef_)
            importances = importances / importances.sum()  # Normalize
        else:
            # No importances available
            importances = np.ones(len(feature_names)) / len(feature_names)

        return dict(zip(feature_names, importances.tolist()))

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Calculate evaluation metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary of metric names to values
        """
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(self._mape(y_true, y_pred)),
        }

    @staticmethod
    def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MAPE as a percentage (0-100)
        """
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train_clv_model(
    profiles: list[CustomerProfile],
    config: TrainingConfig | None = None,
    save_path: str | Path | None = None,
) -> CLVPredictor:
    """Convenience function to train and return a CLV predictor.

    Args:
        profiles: List of CustomerProfile objects for training
        config: Training configuration
        save_path: Optional path to save the model

    Returns:
        Trained CLVPredictor ready for inference
    """
    trainer = CLVTrainer(config)
    result = trainer.train(profiles)
    predictor = trainer.create_predictor(result)

    if save_path:
        predictor.save(save_path)

    return predictor
