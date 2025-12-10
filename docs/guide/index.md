# User Guide

Comprehensive guide to using the Actionable Segmentation Engine effectively.

## Overview

This guide covers:

- **Pipeline mechanics** - How the segmentation pipeline works
- **Working with segments** - Analyzing and filtering results
- **Visualizations** - Creating charts and dashboards
- **Exporting** - Saving results for further analysis

## Chapters

<div class="grid cards" markdown>

-   :material-pipe:{ .lg .middle } **Pipeline Overview**

    ---

    Understand the 11-stage pipeline from data to insights

    [:octicons-arrow-right-24: Pipeline](pipeline.md)

-   :material-account-group:{ .lg .middle } **Working with Segments**

    ---

    Filter, analyze, and understand your customer segments

    [:octicons-arrow-right-24: Segments](segments.md)

-   :material-chart-bar:{ .lg .middle } **Visualizations**

    ---

    Create charts, heatmaps, and dashboards

    [:octicons-arrow-right-24: Visualizations](visualizations.md)

-   :material-export:{ .lg .middle } **Exporting Results**

    ---

    Save results to JSON, CSV, or other formats

    [:octicons-arrow-right-24: Export](export.md)

</div>

## Quick Tips

!!! tip "Start Simple"
    Begin with `quick_segmentation()` before customizing configuration.

!!! tip "Check Production Readiness"
    Use `result.production_ready_segments` to get only validated, actionable segments.

!!! tip "Iterate on Thresholds"
    Validation criteria are configurable - adjust them based on your use case.
