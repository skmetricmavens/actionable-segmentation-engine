# ML-Based CLV Prediction - Implementation Plan

## Executive Summary

This plan extends the existing segmentation engine with ML-based Customer Lifetime Value prediction. It builds on the existing `src/features/` infrastructure rather than creating parallel systems.

## Current Architecture Analysis

### Existing CLV Infrastructure

**`src/features/aggregators.py`** - Current simple CLV calculation:
```python
def calculate_clv_estimate(purchase_metrics, customer_tenure_days, projection_years=3):
    # Monthly frequency × AOV × discounted projection
```
- ⚠️ Projects historical frequency forward (inflates short observation windows)
- ⚠️ No behavioral pattern awareness
- ⚠️ No seasonality handling

**`src/data/schemas.py`** - CustomerProfile already has:
- `clv_estimate`, `churn_risk_score`
- `purchase_frequency_per_month`, `avg_order_value`, `total_revenue`
- `days_since_last_purchase`, `cart_abandonment_rate`
- `category_affinities`, `preferred_day_of_week`, `preferred_hour_of_day`

**`src/features/profile_builder.py`** - Builds profiles using aggregators

### Integration Points

| Component | How CLV Predictor Integrates |
|-----------|------------------------------|
| `profile_builder.py` | Calls `clv_predictor.predict()` instead of `calculate_clv_estimate()` |
| `CustomerProfile` | Add new fields for ML features and prediction metadata |
| `aggregators.py` | Add `aggregate_purchase_intervals()` for new features |
| `trait_extractors.py` | Add `BehaviorTypeExtractor` for regular/irregular classification |
| `clusterer.py` | Use predicted CLV for segment value calculation |

---

## Architecture

### New Module Structure

```
src/
├── features/
│   ├── aggregators.py          # EXTEND: add aggregate_purchase_intervals()
│   ├── profile_builder.py      # MODIFY: use CLVPredictor
│   ├── trait_extractors.py     # EXTEND: add BehaviorTypeExtractor
│   └── clv/                    # NEW: CLV prediction package
│       ├── __init__.py
│       ├── features.py         # Feature engineering for ML
│       ├── predictor.py        # CLVPredictor class
│       ├── training.py         # Model training pipeline
│       ├── explainer.py        # SHAP-based explanations
│       └── models/             # Serialized model storage
├── data/
│   └── schemas.py              # EXTEND: add PurchaseIntervalMetrics, BehaviorType
└── config/
    └── clv_config.py           # NEW: CLV-specific configuration
scripts/
└── train_clv_model.py          # NEW: Training entrypoint
```

---

## Data Model Extensions

### New Fields in `CustomerProfile` (schemas.py)

```python
class PurchaseIntervalMetrics(BaseSchema):
    """Statistics about time between purchases."""
    interval_mean_days: float | None = None       # Average days between purchases
    interval_std_days: float | None = None        # Std dev of intervals
    interval_min_days: float | None = None        # Shortest interval
    interval_max_days: float | None = None        # Longest interval
    interval_cv: float | None = None              # Coefficient of variation
    regularity_index: float | None = None         # 0-1, higher = more regular

class BehaviorType(str, Enum):
    """Customer purchase behavior classification."""
    REGULAR = "regular"           # CV < 0.5, predictable intervals
    IRREGULAR = "irregular"       # CV 0.5-1.0, variable intervals
    LONG_CYCLE = "long_cycle"     # Infrequent but consistent (seasonal)
    NEW = "new"                   # <2 purchases, insufficient data
    ONE_TIME = "one_time"         # Single purchase only

class CustomerProfile(MutableBaseSchema):
    # ... existing fields ...

    # NEW: Purchase interval metrics
    purchase_intervals: PurchaseIntervalMetrics | None = None
    behavior_type: BehaviorType = BehaviorType.NEW

    # NEW: CLV prediction metadata
    clv_predicted: Decimal | None = None          # ML-predicted CLV
    clv_prediction_confidence: float | None = None # Model confidence
    clv_model_version: str | None = None          # Which model made prediction
    clv_top_features: list[str] = Field(default_factory=list)  # Top 3 drivers
```

---

## Task Breakdown

### Phase 1: Data Model & Feature Foundation (Tasks 1-3)

#### Task 1: Add Purchase Interval Aggregation
**File:** `src/features/aggregators.py`

Add function to compute purchase timing statistics:
```python
class PurchaseIntervalMetrics(TypedDict):
    intervals: list[float]         # Days between each purchase
    interval_mean: float | None
    interval_std: float | None
    interval_min: float | None
    interval_max: float | None
    interval_cv: float | None      # Coefficient of variation
    regularity_index: float | None # 1 - normalized_cv

def aggregate_purchase_intervals(events: list[EventRecord]) -> PurchaseIntervalMetrics:
    """Compute statistics about time between purchases."""
```

**Tests:** `tests/test_aggregators.py` - add interval calculation tests

---

#### Task 2: Extend CustomerProfile Schema
**File:** `src/data/schemas.py`

- Add `PurchaseIntervalMetrics` schema
- Add `BehaviorType` enum
- Extend `CustomerProfile` with new fields
- Maintain backward compatibility (all new fields optional)

**Tests:** `tests/test_schemas.py` - validate new fields

---

#### Task 3: Add Behavior Type Classifier
**File:** `src/features/trait_extractors.py`

Add extractor that classifies customers:
```python
class BehaviorTypeExtractor(TraitExtractor):
    """Classify customer as REGULAR, IRREGULAR, LONG_CYCLE, etc."""

    def extract(self, profile: CustomerProfile) -> ActionableTrait | None:
        intervals = profile.purchase_intervals
        if not intervals or profile.total_purchases < 2:
            return None  # NEW or ONE_TIME

        cv = intervals.interval_cv
        if cv < 0.5:
            behavior = BehaviorType.REGULAR
        elif cv < 1.0:
            behavior = BehaviorType.IRREGULAR
        else:
            behavior = BehaviorType.LONG_CYCLE
```

**Thresholds (configurable):**
- REGULAR: CV < 0.5 (predictable timing)
- IRREGULAR: CV 0.5-1.0 (variable timing)
- LONG_CYCLE: CV > 1.0 OR interval_mean > 90 days

---

### Phase 2: ML Feature Engineering (Tasks 4-5)

#### Task 4: Create CLV Feature Builder
**File:** `src/features/clv/features.py` (NEW)

Build ML-ready feature matrix from CustomerProfile:

```python
@dataclass
class CLVFeatureConfig:
    """Configuration for CLV feature engineering."""
    include_rfm: bool = True
    include_engagement: bool = True
    include_temporal: bool = True
    include_intervals: bool = True
    include_seasonality: bool = True
    seasonality_period: int = 365  # days

class CLVFeatureBuilder:
    """Extract ML features from CustomerProfile for CLV prediction."""

    def __init__(self, config: CLVFeatureConfig | None = None):
        self.config = config or CLVFeatureConfig()
        self.feature_names: list[str] = []

    def build_features(self, profile: CustomerProfile) -> dict[str, float]:
        """Extract all features for a single profile."""

    def build_feature_matrix(
        self, profiles: list[CustomerProfile]
    ) -> tuple[np.ndarray, list[str]]:
        """Build feature matrix for batch of profiles."""
```

**Feature Categories:**

| Category | Features |
|----------|----------|
| **RFM Core** | days_since_last_purchase, purchase_frequency_per_month, total_revenue, avg_order_value |
| **Engagement** | total_sessions, total_page_views, cart_abandonment_rate, items_per_session |
| **Intervals** | interval_mean, interval_std, interval_cv, regularity_index |
| **Temporal** | tenure_days, preferred_dow_sin/cos, preferred_hour_sin/cos |
| **Efficiency** | revenue_per_session, purchase_per_session, pages_per_purchase |
| **Category** | top_category_concentration, category_diversity_count |
| **Behavior** | behavior_type (one-hot), churn_risk_score |

**Tests:** `tests/test_clv_features.py`

---

#### Task 5: Add Seasonality Features
**File:** `src/features/clv/features.py`

Extend feature builder with cyclical encodings:
```python
def _add_seasonality_features(self, profile: CustomerProfile, features: dict) -> None:
    """Add sin/cos encoded temporal features."""
    # Month of last purchase
    if profile.last_seen:
        month = profile.last_seen.month
        features["last_month_sin"] = np.sin(2 * np.pi * month / 12)
        features["last_month_cos"] = np.cos(2 * np.pi * month / 12)

    # Day of week preference
    if profile.preferred_day_of_week is not None:
        dow = profile.preferred_day_of_week
        features["pref_dow_sin"] = np.sin(2 * np.pi * dow / 7)
        features["pref_dow_cos"] = np.cos(2 * np.pi * dow / 7)
```

---

### Phase 3: Model Training Pipeline (Tasks 6-8)

#### Task 6: Create CLV Predictor Class
**File:** `src/features/clv/predictor.py` (NEW)

```python
class CLVPredictor:
    """ML-based CLV prediction with explainability."""

    def __init__(
        self,
        model_path: Path | None = None,
        feature_builder: CLVFeatureBuilder | None = None,
    ):
        self.model = None
        self.feature_builder = feature_builder or CLVFeatureBuilder()
        self.model_version: str | None = None
        if model_path:
            self.load(model_path)

    def predict(self, profile: CustomerProfile) -> CLVPrediction:
        """Predict CLV for a single customer."""
        features = self.feature_builder.build_features(profile)
        prediction = self.model.predict([list(features.values())])[0]
        return CLVPrediction(
            clv_predicted=Decimal(str(prediction)),
            confidence=self._calculate_confidence(features),
            top_features=self._get_top_features(features),
            model_version=self.model_version,
        )

    def predict_batch(
        self, profiles: list[CustomerProfile]
    ) -> list[CLVPrediction]:
        """Batch prediction for multiple customers."""

    def load(self, path: Path) -> None:
        """Load trained model from disk."""

    def save(self, path: Path) -> None:
        """Save model to disk."""

@dataclass
class CLVPrediction:
    """Result of CLV prediction."""
    clv_predicted: Decimal
    confidence: float
    top_features: list[str]
    model_version: str
```

**Tests:** `tests/test_clv_predictor.py`

---

#### Task 7: Build Training Pipeline
**File:** `src/features/clv/training.py` (NEW)

```python
@dataclass
class TrainingConfig:
    """Configuration for CLV model training."""
    target_window_days: int = 365          # Predict CLV over this period
    min_history_days: int = 30             # Minimum observation window
    test_size: float = 0.2
    val_size: float = 0.1
    model_type: Literal["lightgbm", "ridge", "ensemble"] = "lightgbm"
    cv_folds: int = 5
    random_state: int = 42

class CLVTrainer:
    """Train and evaluate CLV prediction models."""

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.feature_builder = CLVFeatureBuilder()

    def prepare_training_data(
        self,
        profiles: list[CustomerProfile],
        events: list[EventRecord],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create features (X) and targets (y) for training.

        Target: Actual revenue in next `target_window_days`
        This requires event-level data to compute forward-looking target.
        """

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> CLVPredictor:
        """Train model and return predictor."""

    def evaluate(
        self,
        predictor: CLVPredictor,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainingMetrics:
        """Evaluate model on held-out data."""

@dataclass
class TrainingMetrics:
    """Model performance metrics."""
    r2_score: float
    mae: float
    rmse: float
    mape: float | None  # None if zeros in target
    top_decile_lift: float
    feature_importance: dict[str, float]
```

---

#### Task 8: Time-Aware Data Splitting
**File:** `src/features/clv/training.py`

Ensure temporal integrity - no data leakage:
```python
def temporal_train_test_split(
    events: list[EventRecord],
    profiles: list[CustomerProfile],
    split_date: datetime,
    target_window_days: int = 365,
) -> tuple[TrainData, TestData]:
    """
    Split data by time to prevent leakage.

    Training: Events before split_date
    Validation: Events in [split_date, split_date + target_window]
    Test: Events after target window

    Target for training = actual revenue in validation window
    """
```

---

### Phase 4: Integration & Explainability (Tasks 9-11)

#### Task 9: Integrate with Profile Builder
**File:** `src/features/profile_builder.py`

Modify `build_profile()` to use ML prediction when model available:

```python
def build_profile(
    customer_id: str,
    events: list[EventRecord],
    *,
    merge_map: MergeMap | None = None,
    reference_date: datetime | None = None,
    clv_predictor: CLVPredictor | None = None,  # NEW parameter
) -> CustomerProfile:
    # ... existing aggregation ...

    # Compute purchase intervals (NEW)
    interval_metrics = aggregate_purchase_intervals(events)

    # CLV: Use ML prediction if available, else fallback to simple
    if clv_predictor is not None:
        prediction = clv_predictor.predict(profile)
        clv_estimate = prediction.clv_predicted
        clv_predicted = prediction.clv_predicted
        clv_confidence = prediction.confidence
        clv_top_features = prediction.top_features
    else:
        clv_estimate = calculate_clv_estimate(purchase_metrics, tenure_days)
        clv_predicted = None
        clv_confidence = None
        clv_top_features = []

    return CustomerProfile(
        # ... existing fields ...
        purchase_intervals=interval_metrics,
        clv_estimate=clv_estimate,  # Backward compatible
        clv_predicted=clv_predicted,
        clv_prediction_confidence=clv_confidence,
        clv_top_features=clv_top_features,
    )
```

---

#### Task 10: Add SHAP Explainability
**File:** `src/features/clv/explainer.py` (NEW)

```python
class CLVExplainer:
    """SHAP-based explanation of CLV predictions."""

    def __init__(self, predictor: CLVPredictor):
        self.predictor = predictor
        self.explainer = shap.TreeExplainer(predictor.model)

    def explain(self, profile: CustomerProfile) -> CLVExplanation:
        """Generate explanation for single prediction."""
        features = self.predictor.feature_builder.build_features(profile)
        shap_values = self.explainer.shap_values([list(features.values())])

        return CLVExplanation(
            base_value=self.explainer.expected_value,
            predicted_value=features_to_prediction,
            feature_contributions=dict(zip(feature_names, shap_values[0])),
            top_positive=self._get_top_n(shap_values, positive=True),
            top_negative=self._get_top_n(shap_values, positive=False),
        )

    def generate_narrative(self, explanation: CLVExplanation) -> str:
        """Generate business-friendly explanation text."""
        # Example: "This customer's high CLV is primarily driven by:
        #   1. High purchase frequency (contributing +$450)
        #   2. Strong category loyalty (contributing +$120)
        # Factors reducing CLV: High cart abandonment (-$80)"

@dataclass
class CLVExplanation:
    """Structured explanation of CLV prediction."""
    base_value: float
    predicted_value: float
    feature_contributions: dict[str, float]
    top_positive: list[tuple[str, float]]
    top_negative: list[tuple[str, float]]
    narrative: str | None = None
```

---

#### Task 11: Create Training Script
**File:** `scripts/train_clv_model.py` (NEW)

```python
#!/usr/bin/env python3
"""
Train CLV prediction model.

Usage:
    python scripts/train_clv_model.py --data-dir data/samples/
    python scripts/train_clv_model.py --data-dir data/samples/ --model-type lightgbm
"""
import argparse
from pathlib import Path

from src.data.local_loader import load_events_only
from src.features.clv.training import CLVTrainer, TrainingConfig
from src.features.profile_builder import build_profiles_batch


def main():
    parser = argparse.ArgumentParser(description="Train CLV model")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/clv"))
    parser.add_argument("--model-type", default="lightgbm")
    parser.add_argument("--target-days", type=int, default=365)
    args = parser.parse_args()

    # Load data
    events, id_history = load_events_only(args.data_dir)

    # Build profiles
    profiles = build_profiles_batch(events)

    # Train model
    config = TrainingConfig(
        target_window_days=args.target_days,
        model_type=args.model_type,
    )
    trainer = CLVTrainer(config)

    X, y = trainer.prepare_training_data(profiles, events)
    predictor = trainer.train(X, y)

    # Evaluate and save
    metrics = trainer.evaluate(predictor, X_test, y_test)
    print(f"R² Score: {metrics.r2_score:.3f}")
    print(f"MAE: ${metrics.mae:.2f}")

    predictor.save(args.output_dir / "clv_model.joblib")


if __name__ == "__main__":
    main()
```

---

### Phase 5: Testing & Documentation (Tasks 12-13)

#### Task 12: Comprehensive Test Suite
**Files:**
- `tests/test_clv_features.py` - Feature engineering
- `tests/test_clv_predictor.py` - Prediction logic
- `tests/test_clv_training.py` - Training pipeline
- `tests/test_clv_explainer.py` - SHAP explanations
- `tests/test_clv_integration.py` - End-to-end integration

**Test cases:**
- Edge cases: single purchase, no purchases, very long tenure
- Behavior type classification boundaries
- Feature computation correctness
- Model serialization/deserialization
- Backward compatibility with existing pipeline

---

#### Task 13: Update Pipeline Integration
**File:** `src/pipeline.py`

Add optional CLV prediction to pipeline:
```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    use_ml_clv: bool = False
    clv_model_path: Path | None = None

def run_pipeline(config: PipelineConfig) -> PipelineResult:
    # ... existing steps ...

    # Load CLV predictor if configured
    clv_predictor = None
    if config.use_ml_clv and config.clv_model_path:
        clv_predictor = CLVPredictor(config.clv_model_path)

    # Build profiles with optional ML CLV
    profiles = build_profiles_batch(
        events,
        merge_map=merge_map,
        clv_predictor=clv_predictor,  # NEW
    )
```

---

## Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    # Existing...
    "scikit-learn>=1.3",
    "lightgbm>=4.0",
    "shap>=0.42",
    "joblib>=1.3",
]

[project.optional-dependencies]
clv = [
    "optuna>=3.0",        # Hyperparameter tuning (optional)
]
```

---

## Success Criteria

### Technical
| Metric | Target |
|--------|--------|
| R² Score | ≥ 0.20 on held-out test |
| MAE Improvement | ≥ 15% vs simple baseline |
| Prediction latency | < 5ms per customer |
| All existing tests | Still passing |

### Business
| Requirement | Validation |
|-------------|-----------|
| Backward compatible | `clv_estimate` still works |
| Explainable | Top 3 drivers per prediction |
| Handles edge cases | New customers, one-timers |
| Behavior-aware | Different patterns for regular vs irregular |

---

## Task Dependencies

```
Task 1 (Intervals) ─┬─► Task 3 (Behavior Type)
                    │
Task 2 (Schema) ────┼─► Task 4 (Feature Builder) ─► Task 5 (Seasonality)
                    │                                      │
                    └─► Task 6 (Predictor) ◄───────────────┘
                                │
                                ▼
                        Task 7 (Training) ─► Task 8 (Splitting)
                                │
                                ▼
                        Task 9 (Integration)
                                │
                        ┌───────┴───────┐
                        ▼               ▼
                Task 10 (SHAP)    Task 11 (Script)
                        │               │
                        └───────┬───────┘
                                ▼
                        Task 12 (Tests)
                                │
                                ▼
                        Task 13 (Pipeline)
```

---

## Implementation Notes

### Backward Compatibility
- All new `CustomerProfile` fields have defaults
- Existing `clv_estimate` field preserved (uses ML or fallback)
- Pipeline works with or without trained model
- No breaking changes to existing tests

### Configuration
Create `src/config/clv_config.py`:
```python
@dataclass
class CLVConfig:
    # Feature engineering
    include_seasonality: bool = True
    seasonality_period_days: int = 365

    # Behavior classification thresholds
    regular_cv_threshold: float = 0.5
    long_cycle_interval_days: int = 90

    # Training
    target_window_days: int = 365
    min_purchases_for_training: int = 2

    # Prediction
    fallback_to_simple: bool = True
    model_path: Path | None = None
```

### Graceful Degradation
```python
def get_clv_estimate(profile: CustomerProfile, predictor: CLVPredictor | None) -> Decimal:
    """Get CLV with graceful fallback."""
    if predictor is not None:
        try:
            return predictor.predict(profile).clv_predicted
        except Exception:
            pass  # Fall through to simple method

    # Fallback to simple calculation
    return profile.clv_estimate  # Already computed during profile build
```
