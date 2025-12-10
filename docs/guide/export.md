# Exporting Results

Save segmentation results for further analysis or integration with other systems.

## JSON Export

### Full Results

```python
import json
from src.pipeline import export_results_to_dict

result = run_pipeline(config)
data = export_results_to_dict(result)

with open("segmentation_results.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Export Structure

```json
{
  "success": true,
  "duration_ms": 823.5,
  "n_profiles": 500,
  "n_segments": 5,
  "segments": [
    {
      "segment_id": "segment_0",
      "name": "High-Value Loyalists",
      "size": 85,
      "total_clv": "42500.00",
      "avg_clv": "500.00",
      "defining_traits": ["high_revenue", "frequent_purchases"]
    }
  ],
  "robustness_scores": {
    "segment_0": {
      "feature_stability": 0.75,
      "time_consistency": 0.82,
      "overall_robustness": 0.78
    }
  }
}
```

## CSV Export

### Segments to CSV

```python
import csv

with open("segments.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["segment_id", "name", "size", "total_clv", "avg_clv"])

    for segment in result.segments:
        writer.writerow([
            segment.segment_id,
            segment.name,
            segment.size,
            segment.total_clv,
            segment.avg_clv,
        ])
```

### Customer-Segment Mapping

```python
with open("customer_segments.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["customer_id", "segment_id", "segment_name"])

    for segment in result.segments:
        for customer_id in segment.customer_ids:
            writer.writerow([customer_id, segment.segment_id, segment.name])
```

## Pandas DataFrame

### Segments DataFrame

```python
import pandas as pd

df = pd.DataFrame([{
    "segment_id": s.segment_id,
    "name": s.name,
    "size": s.size,
    "total_clv": float(s.total_clv),
    "avg_clv": float(s.avg_clv),
    "avg_order_value": float(s.avg_order_value),
    "robustness": result.robustness_scores[s.segment_id].overall_robustness,
    "is_actionable": result.actionability_evaluations[s.segment_id].is_actionable,
} for s in result.segments])

print(df)
df.to_csv("segments_analysis.csv", index=False)
```

### Profiles DataFrame

```python
profiles_df = pd.DataFrame([{
    "customer_id": p.customer_id,
    "total_revenue": float(p.total_revenue),
    "purchase_count": p.purchase_count,
    "avg_order_value": float(p.avg_order_value),
    "clv_estimate": float(p.clv_estimate),
} for p in result.profiles])

profiles_df.to_csv("customer_profiles.csv", index=False)
```

## Report Export

### Text Summary

```python
from src.pipeline import format_pipeline_summary

summary = format_pipeline_summary(result)

with open("pipeline_summary.txt", "w") as f:
    f.write(summary)
```

### Full Report

```python
from src.reporting.segment_reporter import report_to_text

if result.report:
    text_report = report_to_text(result.report)
    with open("full_report.txt", "w") as f:
        f.write(text_report)
```

## Database Export

### SQLite

```python
import sqlite3

conn = sqlite3.connect("segmentation.db")
cursor = conn.cursor()

# Create tables
cursor.execute("""
    CREATE TABLE IF NOT EXISTS segments (
        segment_id TEXT PRIMARY KEY,
        name TEXT,
        size INTEGER,
        total_clv REAL,
        avg_clv REAL
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS customer_segments (
        customer_id TEXT,
        segment_id TEXT,
        FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
    )
""")

# Insert data
for segment in result.segments:
    cursor.execute(
        "INSERT INTO segments VALUES (?, ?, ?, ?, ?)",
        (segment.segment_id, segment.name, segment.size,
         float(segment.total_clv), float(segment.avg_clv))
    )

    for customer_id in segment.customer_ids:
        cursor.execute(
            "INSERT INTO customer_segments VALUES (?, ?)",
            (customer_id, segment.segment_id)
        )

conn.commit()
conn.close()
```

## Visualization Export

### Save All Charts

```python
from src.reporting import save_figure, plot_segment_distribution
from src.reporting.segment_reporter import segment_to_summary

summaries = [segment_to_summary(s) for s in result.segments]

# Save segment distribution
fig = plot_segment_distribution(summaries, by="size")
save_figure(fig, "output/segment_sizes.png", dpi=150)

fig = plot_segment_distribution(summaries, by="clv")
save_figure(fig, "output/segment_clv.png", dpi=150)
```

## API Integration

### REST API Payload

```python
def create_api_payload(result):
    return {
        "meta": {
            "generated_at": datetime.utcnow().isoformat(),
            "n_customers": len(result.profiles),
            "n_segments": len(result.segments),
        },
        "segments": [
            {
                "id": s.segment_id,
                "name": s.name,
                "size": s.size,
                "metrics": {
                    "total_clv": str(s.total_clv),
                    "avg_clv": str(s.avg_clv),
                },
                "quality": {
                    "robustness": result.robustness_scores[s.segment_id].overall_robustness,
                    "is_actionable": result.actionability_evaluations[s.segment_id].is_actionable,
                },
            }
            for s in result.production_ready_segments
        ]
    }
```

## Best Practices

!!! tip "Export Production-Ready Only"
    For downstream systems, export only `result.production_ready_segments` to ensure quality.

!!! tip "Include Metadata"
    Always include generation timestamp, configuration used, and quality metrics.

!!! tip "Use Appropriate Precision"
    Convert `Decimal` to appropriate precision for your use case.
