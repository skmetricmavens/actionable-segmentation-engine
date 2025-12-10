# Visualizations

Create charts, heatmaps, and dashboards to understand your segmentation results.

## Setup

```python
from src.reporting import (
    set_style,
    plot_segment_distribution,
    plot_segment_sizes_pie,
    plot_robustness_scores,
    plot_robustness_heatmap,
    plot_actionability_dimensions,
    plot_actionability_by_segment,
    plot_viability_scores,
    plot_segment_dashboard,
    plot_report_summary,
    show_figure,
    save_figure,
)
from src.reporting.segment_reporter import segment_to_summary

# Set visual style
set_style("whitegrid")

# Prepare data
result = quick_segmentation(n_customers=500, n_clusters=5)
summaries = [segment_to_summary(seg) for seg in result.segments]
```

## Distribution Charts

### Segment Sizes

```python
fig = plot_segment_distribution(summaries, by="size", title="Customer Count by Segment")
show_figure(fig)
```

### Total CLV

```python
fig = plot_segment_distribution(summaries, by="clv", title="Total CLV by Segment")
show_figure(fig)
```

### Average CLV

```python
fig = plot_segment_distribution(summaries, by="avg_clv", title="Average CLV by Segment")
show_figure(fig)
```

### Average Order Value

```python
fig = plot_segment_distribution(summaries, by="aov", title="AOV by Segment")
show_figure(fig)
```

## Pie Charts

```python
fig = plot_segment_sizes_pie(summaries, title="Segment Distribution")
show_figure(fig)
```

## Robustness Visualizations

### Bar Chart

```python
if result.robustness_scores:
    fig = plot_robustness_scores(result.robustness_scores, title="Robustness Scores")
    show_figure(fig)
```

### Heatmap

Shows all robustness components:

```python
if result.robustness_scores:
    fig = plot_robustness_heatmap(result.robustness_scores, title="Robustness Heatmap")
    show_figure(fig)
```

## Actionability Visualizations

### Dimensions Distribution

```python
if result.actionability_evaluations:
    fig = plot_actionability_dimensions(
        result.actionability_evaluations,
        title="Actionability Dimensions"
    )
    show_figure(fig)
```

### By Segment

```python
if result.actionability_evaluations:
    fig = plot_actionability_by_segment(
        result.actionability_evaluations,
        title="Actionability by Segment"
    )
    show_figure(fig)
```

## Viability Scores

```python
if result.viabilities:
    fig = plot_viability_scores(result.viabilities, title="Viability Scores")
    show_figure(fig)
```

## Dashboard Views

### Single Segment Dashboard

```python
segment = result.segments[0]

fig = plot_segment_dashboard(
    segment,
    robustness=result.robustness_scores.get(segment.segment_id),
    viability=result.viabilities.get(segment.segment_id),
    actionability=result.actionability_evaluations.get(segment.segment_id),
)
show_figure(fig)
```

### Report Summary

```python
if result.report:
    fig = plot_report_summary(result.report, title="Segmentation Summary")
    show_figure(fig)
```

## Customization

### Figure Size

```python
fig = plot_segment_distribution(
    summaries,
    by="size",
    figsize=(14, 6),  # Width, height in inches
)
```

### Color Palette

```python
fig = plot_segment_sizes_pie(
    summaries,
    color_palette="Set2",  # Any seaborn palette
)
```

### Styles

```python
# Available styles: whitegrid, darkgrid, white, dark, ticks
set_style("darkgrid")
fig = plot_segment_distribution(summaries, by="size")
```

## Saving Figures

### PNG (Raster)

```python
fig = plot_segment_distribution(summaries, by="clv")
save_figure(fig, "segment_clv.png", dpi=150)
```

### PDF (Vector)

```python
save_figure(fig, "segment_clv.pdf")
```

### SVG (Vector)

```python
save_figure(fig, "segment_clv.svg")
```

## Jupyter Notebooks

For interactive exploration, see the example notebooks:

| Notebook | Description |
|----------|-------------|
| `01_quick_start_demo.ipynb` | Basic visualizations |
| `03_visualization_gallery.ipynb` | All chart types |

## Custom Visualizations

### Using Matplotlib Directly

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

# Your custom visualization
sizes = [s.size for s in result.segments]
names = [s.name for s in result.segments]
ax.barh(names, sizes)
ax.set_xlabel("Customers")
ax.set_title("Custom Segment Chart")

plt.tight_layout()
plt.show()
```

### Using Seaborn

```python
import seaborn as sns
import pandas as pd

df = pd.DataFrame([{
    'name': s.name,
    'size': s.size,
    'avg_clv': float(s.avg_clv),
} for s in result.segments])

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='size', y='avg_clv', hue='name', ax=ax)
plt.show()
```
