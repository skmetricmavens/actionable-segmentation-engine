#!/usr/bin/env python3
"""Visualization example for the Actionable Segmentation Engine.

This example demonstrates how to create visualizations.

Usage:
    python examples/visualizations.py
"""

from pathlib import Path

from src.pipeline import quick_segmentation
from src.reporting import (
    set_style,
    plot_segment_distribution,
    plot_segment_sizes_pie,
    plot_robustness_scores,
    plot_robustness_heatmap,
    save_figure,
)
from src.reporting.segment_reporter import segment_to_summary


def main() -> None:
    """Create and save visualizations."""
    print("=" * 60)
    print("Visualization Example")
    print("=" * 60)

    # Run segmentation
    print("\nRunning pipeline...")
    result = quick_segmentation(n_customers=500, n_clusters=5, seed=42)

    # Set style
    set_style("whitegrid")

    # Create output directory
    output_dir = Path("output/charts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    summaries = [segment_to_summary(seg) for seg in result.segments]

    # Create visualizations
    print("\nCreating visualizations...")

    # 1. Segment size distribution
    fig = plot_segment_distribution(summaries, by="size", title="Customer Count by Segment")
    save_figure(fig, str(output_dir / "segment_sizes.png"), dpi=150)
    print(f"  Saved: segment_sizes.png")

    # 2. Total CLV by segment
    fig = plot_segment_distribution(summaries, by="clv", title="Total CLV by Segment")
    save_figure(fig, str(output_dir / "segment_clv.png"), dpi=150)
    print(f"  Saved: segment_clv.png")

    # 3. Average CLV by segment
    fig = plot_segment_distribution(summaries, by="avg_clv", title="Average CLV by Segment")
    save_figure(fig, str(output_dir / "segment_avg_clv.png"), dpi=150)
    print(f"  Saved: segment_avg_clv.png")

    # 4. Pie chart
    fig = plot_segment_sizes_pie(summaries, title="Segment Size Distribution")
    save_figure(fig, str(output_dir / "segment_pie.png"), dpi=150)
    print(f"  Saved: segment_pie.png")

    # 5. Robustness scores
    if result.robustness_scores:
        fig = plot_robustness_scores(result.robustness_scores, title="Segment Robustness Scores")
        save_figure(fig, str(output_dir / "robustness_scores.png"), dpi=150)
        print(f"  Saved: robustness_scores.png")

        # 6. Robustness heatmap
        fig = plot_robustness_heatmap(result.robustness_scores, title="Robustness Components Heatmap")
        save_figure(fig, str(output_dir / "robustness_heatmap.png"), dpi=150)
        print(f"  Saved: robustness_heatmap.png")

    # Summary
    print("\n" + "=" * 60)
    print("VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"  Charts saved to: {output_dir.absolute()}")
    print(f"  Total charts: 6")
    print("\nTo view charts, open the PNG files in your image viewer.")


if __name__ == "__main__":
    main()
