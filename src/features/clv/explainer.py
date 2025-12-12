"""
Module: explainer

Purpose: SHAP-based explanations for CLV predictions.

Provides model-agnostic explanations for CLV predictions using SHAP
(SHapley Additive exPlanations) values to identify which features
drive individual predictions.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.schemas import CustomerProfile
from src.features.clv.features import CLVFeatureBuilder, CLVFeatureConfig

logger = logging.getLogger(__name__)

# SHAP is an optional dependency
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None  # type: ignore


@dataclass
class FeatureContribution:
    """Contribution of a single feature to a prediction."""

    feature_name: str
    feature_value: float
    shap_value: float
    contribution_direction: str  # "positive" or "negative"


@dataclass
class CLVExplanation:
    """Explanation for a CLV prediction."""

    customer_id: str
    base_value: float  # Expected value (average prediction)
    predicted_value: float
    contributions: list[FeatureContribution]

    @property
    def top_positive_features(self) -> list[FeatureContribution]:
        """Get features that increased the prediction."""
        return sorted(
            [c for c in self.contributions if c.shap_value > 0],
            key=lambda x: x.shap_value,
            reverse=True,
        )

    @property
    def top_negative_features(self) -> list[FeatureContribution]:
        """Get features that decreased the prediction."""
        return sorted(
            [c for c in self.contributions if c.shap_value < 0],
            key=lambda x: x.shap_value,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert explanation to dictionary."""
        return {
            "customer_id": self.customer_id,
            "base_value": self.base_value,
            "predicted_value": self.predicted_value,
            "contributions": [
                {
                    "feature": c.feature_name,
                    "value": c.feature_value,
                    "shap_value": c.shap_value,
                    "direction": c.contribution_direction,
                }
                for c in self.contributions
            ],
        }


class CLVExplainer:
    """Generate SHAP-based explanations for CLV predictions.

    Uses TreeExplainer for tree-based models (GradientBoosting, RandomForest)
    or KernelExplainer as a fallback for other model types.

    Example:
        >>> explainer = CLVExplainer(model, feature_builder)
        >>> explainer.fit(training_profiles)
        >>> explanation = explainer.explain(customer_profile)
        >>> for contrib in explanation.top_positive_features[:5]:
        ...     print(f"{contrib.feature_name}: {contrib.shap_value:.2f}")
    """

    def __init__(
        self,
        model: Any,
        feature_builder: CLVFeatureBuilder,
        *,
        use_tree_explainer: bool = True,
        n_background_samples: int = 100,
    ) -> None:
        """Initialize explainer.

        Args:
            model: Trained sklearn model
            feature_builder: Feature builder for extracting features
            use_tree_explainer: Try TreeExplainer first for tree models
            n_background_samples: Number of samples for KernelExplainer background
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is required for CLVExplainer. "
                "Install it with: pip install shap"
            )

        self._model = model
        self._feature_builder = feature_builder
        self._use_tree_explainer = use_tree_explainer
        self._n_background_samples = n_background_samples
        self._explainer: Any = None
        self._feature_names: list[str] = []
        self._is_fitted = False

    def fit(self, profiles: list[CustomerProfile]) -> "CLVExplainer":
        """Fit the explainer using background data.

        For TreeExplainer, this simply creates the explainer.
        For KernelExplainer, this samples background data.

        Args:
            profiles: List of CustomerProfile objects for background

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting explainer with {len(profiles)} background samples")

        # Build feature matrix
        X, feature_names = self._feature_builder.build_feature_matrix(profiles)
        self._feature_names = feature_names

        # Try TreeExplainer for tree-based models
        if self._use_tree_explainer:
            try:
                self._explainer = shap.TreeExplainer(self._model)
                self._is_fitted = True
                logger.info("Using TreeExplainer for SHAP values")
                return self
            except Exception as e:
                logger.info(f"TreeExplainer failed, falling back to KernelExplainer: {e}")

        # Fall back to KernelExplainer
        # Sample background data if needed
        if len(X) > self._n_background_samples:
            indices = np.random.choice(
                len(X), self._n_background_samples, replace=False
            )
            background = X[indices]
        else:
            background = X

        self._explainer = shap.KernelExplainer(self._model.predict, background)
        self._is_fitted = True
        logger.info("Using KernelExplainer for SHAP values")

        return self

    def explain(self, profile: CustomerProfile) -> CLVExplanation:
        """Generate explanation for a single profile.

        Args:
            profile: CustomerProfile to explain

        Returns:
            CLVExplanation with feature contributions

        Raises:
            RuntimeError: If explainer not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Explainer not fitted. Call fit() first.")

        # Build feature vector
        features = self._feature_builder.build_features(profile)
        feature_vector = np.array([
            [features.get(name, 0.0) for name in self._feature_names]
        ])

        # Get SHAP values
        shap_values = self._explainer.shap_values(feature_vector)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]

        # Get expected value
        if hasattr(self._explainer, "expected_value"):
            base_value = self._explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
            else:
                base_value = float(base_value)
        else:
            base_value = 0.0

        # Get predicted value
        predicted = self._model.predict(feature_vector)[0]

        # Build contributions
        contributions = []
        for i, name in enumerate(self._feature_names):
            value = features.get(name, 0.0)
            shap_val = float(shap_values[i])
            contributions.append(
                FeatureContribution(
                    feature_name=name,
                    feature_value=value,
                    shap_value=shap_val,
                    contribution_direction="positive" if shap_val > 0 else "negative",
                )
            )

        return CLVExplanation(
            customer_id=profile.internal_customer_id,
            base_value=base_value,
            predicted_value=float(predicted),
            contributions=contributions,
        )

    def explain_batch(
        self,
        profiles: list[CustomerProfile],
    ) -> list[CLVExplanation]:
        """Generate explanations for multiple profiles.

        Args:
            profiles: List of CustomerProfile objects

        Returns:
            List of CLVExplanation objects
        """
        return [self.explain(p) for p in profiles]

    def get_feature_importance(
        self,
        profiles: list[CustomerProfile],
    ) -> dict[str, float]:
        """Calculate global feature importance using mean absolute SHAP values.

        Args:
            profiles: List of CustomerProfile objects

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._is_fitted:
            raise RuntimeError("Explainer not fitted. Call fit() first.")

        # Build feature matrix
        X, _ = self._feature_builder.build_feature_matrix(profiles)

        # Get SHAP values for all samples
        shap_values = self._explainer.shap_values(X)

        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Calculate mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        return dict(zip(self._feature_names, mean_abs_shap.tolist()))


def create_explainer(
    model: Any,
    config: CLVFeatureConfig | None = None,
) -> CLVExplainer:
    """Convenience function to create a CLV explainer.

    Args:
        model: Trained sklearn model
        config: Feature configuration (uses defaults if None)

    Returns:
        CLVExplainer instance (not fitted)
    """
    feature_builder = CLVFeatureBuilder(config)
    return CLVExplainer(model, feature_builder)
