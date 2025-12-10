# API Reference

Auto-generated API documentation from source code docstrings.

## Modules

<div class="grid cards" markdown>

-   :material-pipe:{ .lg .middle } **Pipeline**

    ---

    Main entry points and configuration

    [:octicons-arrow-right-24: Pipeline](pipeline.md)

-   :material-database:{ .lg .middle } **Data Layer**

    ---

    Schemas, ID resolution, data generation

    [:octicons-arrow-right-24: Data](data.md)

-   :material-chart-timeline-variant:{ .lg .middle } **Features Layer**

    ---

    Profile building and aggregation

    [:octicons-arrow-right-24: Features](features.md)

-   :material-account-group:{ .lg .middle } **Segmentation Layer**

    ---

    Clustering, sensitivity, validation

    [:octicons-arrow-right-24: Segmentation](segmentation.md)

-   :material-robot:{ .lg .middle } **LLM Layer**

    ---

    Actionability and explanation

    [:octicons-arrow-right-24: LLM](llm.md)

-   :material-file-chart:{ .lg .middle } **Reporting Layer**

    ---

    Reports and visualizations

    [:octicons-arrow-right-24: Reporting](reporting.md)

</div>

## Quick Links

### Entry Points

- [`run_pipeline()`](pipeline.md#run_pipeline) - Main pipeline runner
- [`quick_segmentation()`](pipeline.md#quick_segmentation) - One-liner convenience
- [`PipelineConfig`](pipeline.md#PipelineConfig) - Configuration options

### Key Classes

- [`CustomerProfile`](data.md#CustomerProfile) - Aggregated customer data
- [`Segment`](data.md#Segment) - Customer segment
- [`RobustnessScore`](data.md#RobustnessScore) - Stability metrics

### Utilities

- [`export_results_to_dict()`](pipeline.md#export_results_to_dict) - JSON export
- [`format_pipeline_summary()`](pipeline.md#format_pipeline_summary) - Text summary
