# Actionable Segmentation Engine

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://skmetricmavens.github.io/actionable-segmentation-engine/)
[![Tests](https://img.shields.io/badge/tests-492%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)](htmlcov/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

ML + LLM-driven customer segmentation with robustness validation for Bloomreach EBQ data.

## Documentation

**Full documentation**: [https://skmetricmavens.github.io/actionable-segmentation-engine/](https://skmetricmavens.github.io/actionable-segmentation-engine/)

| Section | Description |
|---------|-------------|
| [Getting Started](docs/getting-started/index.md) | Installation, quick start, configuration |
| [User Guide](docs/guide/index.md) | Pipeline overview, working with segments |
| [Architecture](docs/architecture/index.md) | System design, data flow, modules |
| [ADRs](docs/adrs/index.md) | Architecture Decision Records |
| [Runbooks](docs/runbooks/index.md) | Troubleshooting, performance tuning |
| [API Reference](docs/reference/index.md) | Auto-generated API documentation |

## Overview

This POC transforms raw Bloomreach EBQ data into commercially exploitable customer insights using a hybrid ML + LLM architecture. It discovers actionable customer segments tied to revenue, retention, and satisfaction with specific business plays.

## Features

- **Synthetic Bloomreach EBQ data generation** - Test pipeline without real customer data
- **Customer ID merge resolution** - Correctly handles cross-device customer unification
- **Actionable trait extraction** - No PCA/embeddings, only business-interpretable traits
- **ML-based segmentation** - KMeans clustering with automatic k-selection
- **Robustness validation** - Feature and time window sensitivity tests
- **LLM-powered actionability filtering** - Rejects non-actionable segments
- **Business-language insights** - Confidence levels and recommended plays
- **Comprehensive reporting** - JSON export and visualization support

## Installation

```bash
# Clone the repository
git clone https://github.com/skmetricmavens/actionable-segmentation-engine.git
cd actionable-segmentation-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### One-Liner Segmentation

```python
from src.pipeline import quick_segmentation

# Run with synthetic data
result = quick_segmentation(n_customers=500, n_clusters=5, seed=42)

print(f"Generated {len(result.segments)} segments")
print(f"Production-ready: {len(result.production_ready_segments)}")
```

### Full Pipeline with Configuration

```python
from src.pipeline import PipelineConfig, run_pipeline, format_pipeline_summary

# Configure the pipeline
config = PipelineConfig(
    n_customers=1000,
    n_clusters=5,
    auto_select_k=False,
    run_sensitivity=True,
    generate_report=True,
)

# Run pipeline
result = run_pipeline(config)

# Print summary
print(format_pipeline_summary(result))

# Access segments
for segment in result.segments:
    print(f"\n{segment.name}:")
    print(f"  Size: {segment.size} customers")
    print(f"  Total CLV: ${float(segment.total_clv):,.2f}")

    # Get explanation
    explanation = result.explanations.get(segment.segment_id)
    if explanation:
        print(f"  Campaign: {explanation.recommended_campaign}")
```

### Filter Production-Ready Segments

```python
# Get only segments that are valid AND actionable
for segment in result.production_ready_segments:
    robustness = result.robustness_scores[segment.segment_id]
    print(f"{segment.name}: robustness={robustness.overall_robustness:.2f}")
```

## Pipeline Output

The `PipelineResult` contains:

| Property | Description |
|----------|-------------|
| `profiles` | Customer profiles with aggregated metrics |
| `segments` | ML-generated customer segments |
| `robustness_scores` | Per-segment robustness metrics |
| `viabilities` | Economic viability assessments |
| `actionability_evaluations` | LLM actionability results |
| `explanations` | Business-language explanations |
| `report` | Complete segmentation report |

### Convenience Properties

- `result.valid_segments` - Segments passing validation criteria
- `result.actionable_segments` - Segments with actionable dimensions
- `result.production_ready_segments` - Valid AND actionable segments

## Visualizations

```python
from src.reporting import (
    set_style,
    plot_segment_distribution,
    plot_robustness_scores,
    show_figure,
)
from src.reporting.segment_reporter import segment_to_summary

# Set style
set_style("whitegrid")

# Create summaries
summaries = [segment_to_summary(seg) for seg in result.segments]

# Plot segment sizes
fig = plot_segment_distribution(summaries, by="size")
show_figure(fig)

# Plot robustness
fig = plot_robustness_scores(result.robustness_scores)
show_figure(fig)
```

## Project Structure

```
actionable-segmentation-engine/
├── src/
│   ├── data/              # Data schemas, synthetic generation, ID merge
│   │   ├── schemas.py     # Pydantic models
│   │   ├── synthetic_generator.py
│   │   └── joiner.py      # Customer ID merge resolution
│   ├── features/          # Profile building, aggregation
│   │   ├── profile_builder.py
│   │   └── aggregators.py
│   ├── segmentation/      # Clustering, validation, sensitivity
│   │   ├── clusterer.py   # KMeans clustering
│   │   ├── segment_validator.py
│   │   └── sensitivity.py # Robustness analysis
│   ├── llm/               # LLM integration
│   │   ├── actionability_filter.py
│   │   └── segment_explainer.py
│   ├── reporting/         # Reports and visualizations
│   │   ├── segment_reporter.py
│   │   └── visuals.py
│   ├── pipeline.py        # End-to-end orchestrator
│   └── exceptions.py      # Custom exceptions
├── tests/                 # 492+ tests with 94% coverage
├── notebooks/             # Jupyter examples
│   ├── 01_quick_start_demo.ipynb
│   ├── 02_custom_configuration.ipynb
│   └── 03_visualization_gallery.ipynb
└── config/                # Configuration files
```

## Testing

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with verbose output
pytest -v --tb=short

# Type checking
mypy src/ --ignore-missing-imports
```

Current status: **492 tests passing with 94% coverage**

## Configuration Options

### PipelineConfig

| Option | Default | Description |
|--------|---------|-------------|
| `n_customers` | 1000 | Number of synthetic customers |
| `data_seed` | 42 | Random seed for data generation |
| `n_clusters` | None | Fixed cluster count (None = auto) |
| `auto_select_k` | True | Enable automatic k selection |
| `k_range` | (3, 10) | Range for k selection |
| `run_sensitivity` | True | Run robustness analysis |
| `generate_report` | True | Generate full report |
| `validation_criteria` | None | Custom validation criteria |
| `use_llm` | False | Use real LLM (False = mock) |
| `verbose` | False | Print progress output |

### ValidationCriteria

```python
from decimal import Decimal
from src.segmentation.segment_validator import ValidationCriteria

criteria = ValidationCriteria(
    min_segment_size=20,
    max_segment_size_pct=0.4,
    min_total_clv=Decimal("5000"),
    min_avg_clv=Decimal("100"),
    min_feature_stability=0.5,
    min_overall_robustness=0.6,
    min_expected_roi=1.0,
)
```

## Export Results

```python
import json
from src.pipeline import export_results_to_dict

# Export to dictionary
data = export_results_to_dict(result)

# Save to JSON
with open("results.json", "w") as f:
    json.dump(data, f, indent=2)
```

## Architecture

The pipeline executes 10 stages:

1. **Data Acquisition** - Load or generate event data
2. **ID Resolution** - Resolve customer merge chains
3. **Profile Building** - Aggregate events into profiles
4. **Clustering** - KMeans segmentation
5. **Sensitivity Analysis** - Feature/time stability tests
6. **Robustness Scoring** - Per-segment robustness
7. **Validation** - Apply business criteria
8. **Viability Assessment** - Economic evaluation
9. **Actionability Evaluation** - LLM assessment
10. **Explanation Generation** - Business insights
11. **Report Generation** - Create final report

## Notebooks

See `notebooks/` for interactive examples:

- **01_quick_start_demo.ipynb** - Get started in 5 minutes
- **02_custom_configuration.ipynb** - Advanced configuration
- **03_visualization_gallery.ipynb** - All visualization types

## Building Documentation

```bash
# Install documentation dependencies
pip install mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions

# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

## Backstage Integration

This project includes `catalog-info.yaml` for [Backstage](https://backstage.io/) service catalog integration. The documentation follows Spotify's [TechDocs](https://backstage.io/docs/features/techdocs/) docs-like-code approach.

## License

MIT
