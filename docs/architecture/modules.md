# Module Reference

Detailed documentation of each module in the codebase.

## Project Structure

```
src/
├── data/                  # Data layer
│   ├── schemas.py         # Pydantic models
│   ├── joiner.py          # ID merge resolution
│   └── synthetic_generator.py
├── features/              # Features layer
│   ├── profile_builder.py
│   └── aggregators.py
├── segmentation/          # Segmentation layer
│   ├── clusterer.py
│   ├── sensitivity.py
│   └── segment_validator.py
├── llm/                   # LLM layer
│   ├── actionability_filter.py
│   └── segment_explainer.py
├── reporting/             # Reporting layer
│   ├── segment_reporter.py
│   └── visuals.py
├── pipeline.py            # Orchestrator
└── exceptions.py          # Exception hierarchy
```

## Data Layer

### schemas.py

Pydantic v2 models for all data types.

**Key Models:**

| Model | Purpose |
|-------|---------|
| `EventRecord` | Raw event data |
| `CustomerProfile` | Aggregated metrics |
| `Segment` | Customer segment |
| `RobustnessScore` | Stability metrics |
| `ActionabilityEvaluation` | LLM evaluation |
| `SegmentExplanation` | Business insight |

**Example:**

```python
from src.data.schemas import Segment

segment = Segment(
    segment_id="seg_001",
    name="High-Value",
    size=100,
    customer_ids=["c1", "c2"],
    total_clv=Decimal("50000"),
    avg_clv=Decimal("500"),
    defining_traits=["high_revenue"],
)
```

### joiner.py

Customer ID merge resolution.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `resolve_customer_merges()` | Build merge map from ID history |
| `apply_merge_map()` | Apply merges to event stream |

**Handles:**
- Direct merges (A → B)
- Chain merges (A → B → C)
- Circular detection

### synthetic_generator.py

Realistic test data generation.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `generate_small_dataset()` | Generate complete dataset |
| `generate_events()` | Generate event records |
| `generate_id_history()` | Generate merge history |

**Customer Archetypes:**
- High-value loyalists
- Bargain hunters
- Occasional browsers
- New customers
- Churning customers

## Features Layer

### profile_builder.py

Transforms events into customer profiles.

**Key Class: `ProfileBuilder`**

```python
builder = ProfileBuilder(config)
profiles = builder.build_profiles(events, merge_map)
```

**Profile Metrics:**
- Revenue (total, average, frequency)
- Engagement (sessions, pages)
- Temporal (preferred day/hour)
- Predictive (CLV, churn risk)

### aggregators.py

Individual aggregation functions.

| Function | Output |
|----------|--------|
| `aggregate_purchases()` | Revenue metrics |
| `aggregate_sessions()` | Session behavior |
| `aggregate_temporal()` | Time patterns |
| `aggregate_categories()` | Category preferences |
| `calculate_clv_estimate()` | Lifetime value |

**Design:** Pure functions for easy testing.

## Segmentation Layer

### clusterer.py

KMeans clustering with auto k-selection.

**Key Class: `CustomerClusterer`**

```python
clusterer = CustomerClusterer(
    n_clusters=5,
    auto_select_k=False,
    seed=42,
)
result = clusterer.cluster(profiles)
```

**Features:**
- Automatic feature extraction
- StandardScaler normalization
- Elbow method for k selection
- Silhouette score validation

### sensitivity.py

Robustness testing via sensitivity analysis.

**Key Class: `SensitivityAnalyzer`**

```python
analyzer = SensitivityAnalyzer()
result = analyzer.analyze_segments(profiles, segments)
```

**Tests:**

| Test | Method |
|------|--------|
| Feature Drop | Remove each feature, re-cluster |
| Time Window | Cluster different periods |
| Bootstrap | Re-sample with replacement |

### segment_validator.py

Business validation and viability assessment.

**Key Class: `SegmentValidator`**

```python
validator = SegmentValidator(criteria)
validation = validator.validate(segment, robustness)
viability = validator.assess_viability(segment)
```

**Validation Checks:**
- Size constraints
- CLV thresholds
- Robustness requirements

## LLM Layer

### actionability_filter.py

Evaluates segment actionability.

**Key Class: `ActionabilityFilter`**

```python
filter = ActionabilityFilter(use_mock=True)
evaluation = filter.evaluate(segment)
```

**Dimensions:**
- **WHAT** - Product preferences
- **WHEN** - Timing patterns
- **HOW** - Channel preferences
- **WHO** - Value tier

### segment_explainer.py

Generates business explanations.

**Key Class: `SegmentExplainer`**

```python
explainer = SegmentExplainer(use_mock=True)
explanation = explainer.explain(segment, evaluation)
```

**Output:**
- Executive summary
- Campaign recommendation
- ROI expectation
- Confidence justification

## Reporting Layer

### segment_reporter.py

Report generation and serialization.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `generate_segmentation_report()` | Create full report |
| `segment_to_summary()` | Convert for visualization |
| `report_to_json()` | JSON serialization |

### visuals.py

Matplotlib/seaborn visualizations.

**Key Functions:**

| Function | Chart Type |
|----------|------------|
| `plot_segment_distribution()` | Bar chart |
| `plot_segment_sizes_pie()` | Pie chart |
| `plot_robustness_scores()` | Bar chart |
| `plot_robustness_heatmap()` | Heatmap |
| `plot_segment_dashboard()` | Multi-panel |

## Pipeline Module

### pipeline.py

End-to-end orchestrator.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `PipelineConfig` | Configuration dataclass |
| `PipelineResult` | Result container |
| `run_pipeline()` | Main entry point |
| `quick_segmentation()` | One-liner convenience |

**Convenience Properties:**

```python
result.valid_segments           # Pass validation
result.actionable_segments      # Have actionability
result.production_ready_segments # Both
```

## Exceptions Module

### exceptions.py

Structured exception hierarchy.

```
SegmentationEngineError (base)
├── DataValidationError
├── CustomerMergeError
│   ├── CircularMergeError
│   └── MergeChainTooDeepError
├── ProfileBuildError
├── InsufficientDataError
├── ClusteringError
├── SensitivityTestError
├── SegmentRejectedError
├── LLMIntegrationError
│   └── LLMError
├── ReportGenerationError
└── PipelineError
```
