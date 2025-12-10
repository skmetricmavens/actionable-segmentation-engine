# Configuration

Learn how to customize the segmentation pipeline for your needs.

## PipelineConfig

The `PipelineConfig` dataclass controls all pipeline behavior:

```python
from src.pipeline import PipelineConfig

config = PipelineConfig(
    # Data generation
    n_customers=1000,
    data_seed=42,
    merge_probability=0.15,

    # Clustering
    n_clusters=None,          # None = auto-select
    auto_select_k=True,
    k_range=(3, 10),
    cluster_seed=42,

    # Analysis
    run_sensitivity=True,
    include_sampling_stability=True,

    # Validation
    validation_criteria=None,  # Use defaults

    # Output
    generate_report=True,
    use_llm=False,            # True = real Claude API
    verbose=False,
)
```

## Configuration Options

### Data Generation

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `n_customers` | `int` | 1000 | Number of synthetic customers |
| `data_seed` | `int` | 42 | Random seed for reproducibility |
| `merge_probability` | `float` | 0.15 | Probability of customer ID merges |
| `date_range` | `tuple` | None | Date range for events |

### Clustering

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `n_clusters` | `int \| None` | None | Fixed cluster count (None = auto) |
| `auto_select_k` | `bool` | True | Enable automatic k selection |
| `k_range` | `tuple[int, int]` | (3, 10) | Range for k selection |
| `cluster_seed` | `int` | 42 | Clustering random seed |

### Sensitivity Analysis

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `run_sensitivity` | `bool` | True | Enable robustness analysis |
| `include_sampling_stability` | `bool` | True | Include bootstrap sampling |

### Output

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `generate_report` | `bool` | True | Generate full report |
| `use_llm` | `bool` | False | Use real Claude API |
| `verbose` | `bool` | False | Print progress output |

## ValidationCriteria

Customize segment validation thresholds:

```python
from decimal import Decimal
from src.segmentation.segment_validator import ValidationCriteria

criteria = ValidationCriteria(
    # Size constraints
    min_segment_size=20,
    max_segment_size_pct=0.4,      # Max 40% of customers

    # Value thresholds
    min_total_clv=Decimal("5000"),
    min_avg_clv=Decimal("100"),

    # Robustness requirements
    min_feature_stability=0.5,
    min_overall_robustness=0.6,

    # ROI expectations
    min_expected_roi=1.0,          # Require 100% ROI
)

config = PipelineConfig(
    validation_criteria=criteria,
)
```

## Environment Variables

For LLM integration:

```bash
# Required for use_llm=True
export ANTHROPIC_API_KEY="your-api-key"

# Optional model override
export ANTHROPIC_MODEL="claude-sonnet-4-20250514"
```

## Configuration Patterns

### Quick Testing

```python
config = PipelineConfig(
    n_customers=100,
    n_clusters=3,
    auto_select_k=False,
    run_sensitivity=False,
    generate_report=False,
)
```

### Production Analysis

```python
config = PipelineConfig(
    n_customers=10000,
    auto_select_k=True,
    k_range=(5, 15),
    run_sensitivity=True,
    include_sampling_stability=True,
    generate_report=True,
    use_llm=True,
    verbose=True,
)
```

### Strict Validation

```python
strict_criteria = ValidationCriteria(
    min_segment_size=50,
    min_overall_robustness=0.7,
    min_expected_roi=2.0,
)

config = PipelineConfig(
    validation_criteria=strict_criteria,
)
```

## Next Steps

- [Pipeline Overview](../guide/pipeline.md) - Understand pipeline stages
- [Working with Segments](../guide/segments.md) - Analyze segment results
