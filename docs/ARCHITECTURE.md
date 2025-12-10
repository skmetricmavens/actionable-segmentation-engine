# Architecture Documentation

## Overview

The Actionable Segmentation Engine uses a modular pipeline architecture that transforms raw event data into actionable customer segments with business insights.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE ORCHESTRATOR                              │
│                              (src/pipeline.py)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
    ┌──────────────────────────────────┼──────────────────────────────────┐
    │                                  │                                   │
    ▼                                  ▼                                   ▼
┌────────────┐                  ┌────────────┐                    ┌────────────┐
│   DATA     │                  │  FEATURES  │                    │ SEGMENTATION│
│ LAYER      │                  │   LAYER    │                    │   LAYER     │
│            │                  │            │                    │             │
│ • schemas  │                  │ • profile  │                    │ • clusterer │
│ • joiner   │ ─────────────▶  │   builder  │ ────────────────▶  │ • validator │
│ • synthetic│                  │ • aggregat │                    │ • sensitiv. │
│   generator│                  │            │                    │             │
└────────────┘                  └────────────┘                    └────────────┘
                                                                        │
                              ┌─────────────────────────────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │     LLM LAYER      │
                    │                    │
                    │ • actionability    │
                    │   filter           │
                    │ • segment          │
                    │   explainer        │
                    └────────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   REPORTING LAYER  │
                    │                    │
                    │ • segment_reporter │
                    │ • visuals          │
                    └────────────────────┘
```

## Data Flow

### 1. Data Acquisition
- Input: Raw events, customer ID history
- Output: Validated event records, merge history
- Module: `src/data/`

### 2. ID Resolution
- Input: Customer ID history records
- Output: Merge map (old_id -> canonical_id)
- Module: `src/data/joiner.py`
- Key function: `resolve_customer_merges()`

### 3. Profile Building
- Input: Events, merge map
- Output: Customer profiles with aggregated metrics
- Module: `src/features/profile_builder.py`
- Key class: `ProfileBuilder`

### 4. Clustering
- Input: Customer profiles
- Output: Segments with cluster assignments
- Module: `src/segmentation/clusterer.py`
- Key class: `CustomerClusterer`

### 5. Sensitivity Analysis
- Input: Profiles, segments
- Output: Robustness scores
- Module: `src/segmentation/sensitivity.py`
- Key class: `SensitivityAnalyzer`

### 6. Validation
- Input: Segments, robustness scores
- Output: Validation results, viability assessments
- Module: `src/segmentation/segment_validator.py`
- Key class: `SegmentValidator`

### 7. LLM Processing
- Input: Segments, robustness, viability
- Output: Actionability evaluations, explanations
- Modules: `src/llm/actionability_filter.py`, `src/llm/segment_explainer.py`
- Key classes: `ActionabilityFilter`, `SegmentExplainer`

### 8. Reporting
- Input: All pipeline outputs
- Output: Reports, visualizations
- Module: `src/reporting/`

## Module Details

### Data Layer (`src/data/`)

#### schemas.py
Pydantic v2 models for all data types:
- `EventRecord` - Raw event data
- `CustomerProfile` - Aggregated customer metrics
- `Segment` - Customer segment with traits
- `RobustnessScore` - Segment stability metrics
- `SegmentViability` - Economic viability assessment
- `ActionabilityEvaluation` - LLM evaluation result
- `SegmentExplanation` - Business-language insight

#### joiner.py
Customer ID merge resolution:
- Handles recursive merge chains (A -> B -> C)
- Detects circular merges
- Produces canonical ID mapping

#### synthetic_generator.py
Realistic synthetic data generation:
- Multiple customer behavior archetypes
- Realistic event sequences
- Configurable merge probability

### Features Layer (`src/features/`)

#### profile_builder.py
Aggregates events into customer profiles:
- Revenue metrics (total, average, frequency)
- Engagement metrics (sessions, page views)
- Temporal patterns (preferred days/hours)
- Churn risk scoring
- CLV estimation

#### aggregators.py
Individual aggregation functions:
- `aggregate_purchases()` - Purchase metrics
- `aggregate_sessions()` - Session behavior
- `aggregate_temporal()` - Time patterns
- `aggregate_categories()` - Category preferences
- `calculate_clv_estimate()` - Lifetime value

### Segmentation Layer (`src/segmentation/`)

#### clusterer.py
KMeans clustering with:
- Feature extraction from profiles
- Automatic k-selection (elbow method)
- Segment creation from clusters

#### sensitivity.py
Robustness testing:
- Feature drop stability (leave-one-out)
- Time window stability
- Bootstrap sampling stability

#### segment_validator.py
Business validation:
- Size thresholds
- CLV requirements
- Robustness thresholds
- ROI estimation
- Strategic impact assessment

### LLM Layer (`src/llm/`)

#### actionability_filter.py
Evaluates segment actionability:
- WHAT dimension (product preferences)
- WHEN dimension (timing patterns)
- HOW dimension (channel preferences)
- WHO dimension (customer value)

#### segment_explainer.py
Generates business insights:
- Executive summaries
- Campaign recommendations
- ROI expectations
- Confidence justifications

### Reporting Layer (`src/reporting/`)

#### segment_reporter.py
Report generation:
- Segment reports with all data
- Full segmentation reports
- JSON export/import
- Text summaries

#### visuals.py
Matplotlib/seaborn visualizations:
- Distribution charts
- Pie charts
- Robustness heatmaps
- Actionability plots
- Dashboard views

## Pipeline Configuration

### PipelineConfig

```python
@dataclass
class PipelineConfig:
    # Data generation
    n_customers: int = 1000
    data_seed: int = 42
    merge_probability: float = 0.15

    # Clustering
    n_clusters: int | None = None
    auto_select_k: bool = True
    k_range: tuple[int, int] = (3, 10)

    # Analysis
    run_sensitivity: bool = True
    validation_criteria: ValidationCriteria | None = None

    # Output
    generate_report: bool = True
    use_llm: bool = False
```

### ValidationCriteria

```python
@dataclass
class ValidationCriteria:
    min_segment_size: int = 10
    max_segment_size_pct: float = 0.5
    min_total_clv: Decimal = Decimal("1000")
    min_feature_stability: float = 0.3
    min_overall_robustness: float = 0.4
    min_expected_roi: float = 0.5
```

## Exception Hierarchy

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

## Key Design Decisions

### 1. Pydantic v2 Models
- Strict validation at boundaries
- Immutable base models where appropriate
- Clear serialization support

### 2. Functional Core with Class Wrappers
- Pure functions for core logic
- Classes for stateful operations and configuration
- Easy testing of both approaches

### 3. Mock LLM Support
- MVP uses deterministic mock responses
- Real LLM can be plugged in via protocol
- Consistent interface for testing

### 4. Robustness-First Segmentation
- Segments must prove stability before approval
- Multiple sensitivity tests (feature, time, sampling)
- Production-readiness flag

### 5. Business-Actionable Outputs
- Every segment has clear "what to do"
- ROI estimates for prioritization
- Confidence levels for decision support

## Performance Considerations

### Current Benchmarks (500 customers, 5 clusters)
- Total pipeline: ~800ms
- Data acquisition: ~500ms
- Clustering: ~30ms
- Sensitivity: ~200ms
- LLM evaluation: ~10ms

### Scaling Recommendations
- For 10k+ customers: Consider incremental profiling
- For 100k+ customers: Use batch processing with sampling
- For real LLM: Implement async batch calls
