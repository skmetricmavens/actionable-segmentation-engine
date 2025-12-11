# Quick Start

Get up and running with your first customer segmentation in under 5 minutes.

## One-Liner Segmentation (Synthetic Data)

The fastest way to see the engine in action:

```python
from src.pipeline import quick_segmentation

result = quick_segmentation(n_customers=500, n_clusters=5, seed=42)

print(f"Generated {len(result.segments)} segments")
print(f"Production-ready: {len(result.production_ready_segments)}")
```

## Using Real Data (From BigQuery or Parquet Files)

Load data from local parquet files (extracted from BigQuery or other sources):

```python
from src.data import load_events_only
from src.pipeline import run_pipeline, PipelineConfig

# Load sample data from parquet files
events, id_history = load_events_only("data/samples")

# Configure and run pipeline
config = PipelineConfig(
    min_events_per_customer=3,
    n_clusters=5,
    run_sensitivity=True,
    verbose=True,
)

result = run_pipeline(config, events=events, id_history=id_history)
print(f"Analyzed {len(result.profiles)} customers")
```

### Working with Different Data Sources

The engine supports flexible schema mapping for different data sources:

```python
from src.data import (
    load_events_only,
    create_ga4_config,      # Google Analytics 4
    create_segment_config,  # Segment.com
    create_bloomreach_config,  # Bloomreach (default)
)

# For Bloomreach data (default)
events, id_history = load_events_only("data/bloomreach_export")

# For Google Analytics 4 data
events, id_history = load_events_only(
    "data/ga4_export",
    schema_config=create_ga4_config()
)

# For Segment.com data
events, id_history = load_events_only(
    "data/segment_export",
    schema_config=create_segment_config()
)
```

### Custom Schema Configuration

For custom data sources, create your own configuration:

```python
from src.data import ClientSchemaConfig, load_events_only

# Configure for your data structure
custom_config = ClientSchemaConfig(
    client_name="my_company",
    customer_id_field="user_id",           # Your customer ID field
    timestamp_field="event_time",          # Your timestamp field
    properties_field="event_data",         # Nested properties field (or None if flat)

    # ID merge configuration
    id_history_table="user_id_mapping",
    past_id_field="old_user_id",
    canonical_id_field="current_user_id",

    # Field alternatives (tries each until one has a value)
    category_fields=["category", "product_type", "item_category"],
    revenue_fields=["total", "amount", "transaction_value"],

    # Device detection
    mobile_device_values=["mobile", "iOS", "Android", "tablet"],
)

events, id_history = load_events_only("data/custom_export", schema_config=custom_config)
```

## Understanding the Output

The `PipelineResult` contains everything you need:

```python
# Access all segments
for segment in result.segments:
    print(f"{segment.name}: {segment.size} customers, CLV: ${float(segment.total_clv):,.2f}")

# Get only production-ready segments (valid AND actionable)
for segment in result.production_ready_segments:
    robustness = result.robustness_scores[segment.segment_id]
    print(f"{segment.name}: robustness={robustness.overall_robustness:.2f}")

# View business explanations
for segment in result.segments:
    explanation = result.explanations.get(segment.segment_id)
    if explanation:
        print(f"{segment.name}: {explanation.recommended_campaign}")
```

## Full Pipeline Example

For more control, use `PipelineConfig`:

```python
from src.pipeline import PipelineConfig, run_pipeline, format_pipeline_summary

# Configure the pipeline
config = PipelineConfig(
    n_customers=1000,         # For synthetic data
    n_clusters=5,
    auto_select_k=False,      # Use fixed k
    run_sensitivity=True,     # Enable robustness analysis
    generate_report=True,     # Generate full report
    verbose=True,             # Print progress
)

# Run pipeline
result = run_pipeline(config)

# Print formatted summary
print(format_pipeline_summary(result))
```

## Command-Line Usage

Run the pipeline from the command line:

```bash
# Run on local sample data
python scripts/run_local_pipeline.py --clusters 5 --verbose

# Save results to JSON
python scripts/run_local_pipeline.py --clusters 5 -o outputs/results.json

# Skip sensitivity analysis for faster results
python scripts/run_local_pipeline.py --clusters 5 --no-sensitivity
```

## Visualizing Results

Create visualizations with the reporting module:

```python
from src.reporting import (
    set_style,
    plot_segment_distribution,
    plot_robustness_scores,
    show_figure,
)
from src.reporting.segment_reporter import segment_to_summary

# Set visual style
set_style("whitegrid")

# Create summaries for plotting
summaries = [segment_to_summary(seg) for seg in result.segments]

# Plot segment sizes
fig = plot_segment_distribution(summaries, by="size")
show_figure(fig)

# Plot robustness scores
if result.robustness_scores:
    fig = plot_robustness_scores(result.robustness_scores)
    show_figure(fig)
```

## Exporting Results

Save results for further analysis:

```python
import json
from src.pipeline import export_results_to_dict

# Export to dictionary
data = export_results_to_dict(result)

# Save to JSON
with open("segmentation_results.json", "w") as f:
    json.dump(data, f, indent=2, default=str)
```

## Interactive Notebooks

For a more interactive experience, check out the Jupyter notebooks:

| Notebook | Description |
|----------|-------------|
| `01_quick_start_demo.ipynb` | Get started in 5 minutes |
| `02_local_data_pipeline.ipynb` | Working with real data |
| `03_custom_schema_config.ipynb` | Configure for your data source |
| `04_visualization_gallery.ipynb` | All visualization types |

## Next Steps

- [Configuration Guide](configuration.md) - Customize pipeline behavior
- [Working with Segments](../guide/segments.md) - Deep dive into segment analysis
- [Data Loading](../reference/data.md) - Complete data loading reference
- [API Reference](../reference/index.md) - Complete API documentation
