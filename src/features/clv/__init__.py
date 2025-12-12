"""
CLV Prediction Package.

This package provides ML-based Customer Lifetime Value prediction capabilities.

Modules:
- features: Feature engineering for CLV prediction
- predictor: CLVPredictor class for inference
- training: Training pipeline and utilities
- explainer: SHAP-based prediction explanations
"""

from src.features.clv.features import CLVFeatureBuilder, CLVFeatureConfig
from src.features.clv.predictor import CLVPrediction, CLVPredictor, ModelMetadata
from src.features.clv.training import (
    CLVTrainer,
    TrainingConfig,
    TrainingResult,
    train_clv_model,
)

# Explainer imports are conditional (SHAP is optional)
try:
    from src.features.clv.explainer import (
        CLVExplainer,
        CLVExplanation,
        FeatureContribution,
        create_explainer,
    )

    _EXPLAINER_EXPORTS = [
        "CLVExplainer",
        "CLVExplanation",
        "FeatureContribution",
        "create_explainer",
    ]
except ImportError:
    _EXPLAINER_EXPORTS = []

__all__ = [
    "CLVFeatureBuilder",
    "CLVFeatureConfig",
    "CLVPrediction",
    "CLVPredictor",
    "CLVTrainer",
    "ModelMetadata",
    "TrainingConfig",
    "TrainingResult",
    "train_clv_model",
    *_EXPLAINER_EXPORTS,
]
