# Quick Start

Get up and running with your first customer segmentation in under 5 minutes.

## One-Liner Segmentation

The fastest way to see the engine in action:

```python
from src.pipeline import quick_segmentation

result = quick_segmentation(n_customers=500, n_clusters=5, seed=42)

print(f"Generated {len(result.segments)} segments")
print(f"Production-ready: {len(result.production_ready_segments)}")
```

## Understanding the Output

The `PipelineResult` contains everything you need:

```python
# Access all segments
for segment in result.segments:
    print(f"{segment.name}: {segment.size} customers")

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
    n_customers=1000,
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
    json.dump(data, f, indent=2)
```

## Interactive Notebooks

For a more interactive experience, check out the Jupyter notebooks:

| Notebook | Description |
|----------|-------------|
| `01_quick_start_demo.ipynb` | Get started in 5 minutes |
| `02_custom_configuration.ipynb` | Advanced configuration options |
| `03_visualization_gallery.ipynb` | All visualization types |

## Next Steps

- [Configuration Guide](configuration.md) - Customize pipeline behavior
- [Working with Segments](../guide/segments.md) - Deep dive into segment analysis
- [API Reference](../reference/index.md) - Complete API documentation
