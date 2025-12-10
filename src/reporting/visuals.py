"""
Module: visuals

Purpose: Visualization utilities for segments and analysis.

Key Functions:
- plot_segment_distribution: Visualize segment sizes
- plot_feature_importance: Show feature contributions
- plot_sensitivity_results: Visualize robustness scores

Architecture Notes:
- Uses matplotlib and seaborn
- Returns figure objects for notebook integration
- Optional for MVP but useful for debugging
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from src.data.schemas import (
    ActionabilityDimension,
    ActionabilityEvaluation,
    RobustnessScore,
    RobustnessTier,
    Segment,
    SegmentViability,
)
from src.reporting.segment_reporter import SegmentationReport


# =============================================================================
# CONFIGURATION
# =============================================================================


def set_style(style: str = "whitegrid") -> None:
    """Set the default plotting style."""
    sns.set_style(style)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 10


# =============================================================================
# SEGMENT DISTRIBUTION PLOTS
# =============================================================================


def plot_segment_distribution(
    segments: list[Segment],
    *,
    by: str = "size",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    color_palette: str = "viridis",
) -> Figure:
    """
    Plot segment distribution by size or value.

    Args:
        segments: List of segments to plot
        by: Metric to plot ("size", "clv", "avg_clv", "aov")
        title: Optional custom title
        figsize: Figure size
        color_palette: Seaborn color palette

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data
    names = [s.name for s in segments]
    values: list[float | int]
    if by == "size":
        values = [s.size for s in segments]
        ylabel = "Number of Customers"
        default_title = "Segment Size Distribution"
    elif by == "clv":
        values = [float(s.total_clv) for s in segments]
        ylabel = "Total CLV ($)"
        default_title = "Segment Total CLV Distribution"
    elif by == "avg_clv":
        values = [float(s.avg_clv) for s in segments]
        ylabel = "Average CLV ($)"
        default_title = "Segment Average CLV Distribution"
    else:  # aov
        values = [float(s.avg_order_value) for s in segments]
        ylabel = "Average Order Value ($)"
        default_title = "Segment AOV Distribution"

    # Sort by value
    sorted_pairs = sorted(zip(values, names), reverse=True)
    sorted_values, sorted_names = zip(*sorted_pairs)

    # Create bar plot
    colors = sns.color_palette(color_palette, len(segments))
    bars = ax.barh(range(len(segments)), sorted_values, color=colors)

    # Customize
    ax.set_yticks(range(len(segments)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel(ylabel)
    ax.set_title(title or default_title)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_values)):
        if by in ("clv", "aov", "avg_clv"):
            ax.text(val, i, f" ${val:,.0f}", va="center", fontsize=9)
        else:
            ax.text(val, i, f" {val:,}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_segment_sizes_pie(
    segments: list[Segment],
    *,
    title: str = "Customer Distribution by Segment",
    figsize: tuple[int, int] = (10, 8),
    color_palette: str = "Set2",
) -> Figure:
    """
    Plot segment sizes as a pie chart.

    Args:
        segments: List of segments to plot
        title: Chart title
        figsize: Figure size
        color_palette: Seaborn color palette

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sizes = [s.size for s in segments]
    labels = [s.name for s in segments]
    colors = sns.color_palette(color_palette, len(segments))

    # Create pie chart - unpack result (returns tuple of varying length)
    pie_result = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    # pie_result contains (wedges, texts, autotexts) when autopct is provided
    _ = pie_result  # Suppress unused variable warning

    ax.set_title(title)
    plt.tight_layout()
    return fig


# =============================================================================
# ROBUSTNESS PLOTS
# =============================================================================


def plot_robustness_scores(
    robustness_scores: dict[str, RobustnessScore],
    *,
    title: str = "Segment Robustness Scores",
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """
    Plot robustness scores for segments.

    Args:
        robustness_scores: Mapping of segment_id to RobustnessScore
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Overall robustness bar chart
    segment_ids = list(robustness_scores.keys())
    overall_scores = [r.overall_robustness for r in robustness_scores.values()]

    # Color by tier
    colors = []
    for r in robustness_scores.values():
        if r.robustness_tier == RobustnessTier.HIGH:
            colors.append("green")
        elif r.robustness_tier == RobustnessTier.MEDIUM:
            colors.append("orange")
        else:
            colors.append("red")

    ax1.barh(segment_ids, overall_scores, color=colors)
    ax1.axvline(x=0.7, color="green", linestyle="--", alpha=0.5, label="High threshold")
    ax1.axvline(x=0.4, color="red", linestyle="--", alpha=0.5, label="Low threshold")
    ax1.set_xlabel("Overall Robustness Score")
    ax1.set_title("Overall Robustness by Segment")
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, 1)

    # Right: Breakdown of components
    feature_stability = [r.feature_stability for r in robustness_scores.values()]
    time_consistency = [r.time_window_consistency for r in robustness_scores.values()]

    x = np.arange(len(segment_ids))
    width = 0.35

    ax2.bar(x - width / 2, feature_stability, width, label="Feature Stability")
    ax2.bar(x + width / 2, time_consistency, width, label="Time Consistency")
    ax2.set_xlabel("Segment")
    ax2.set_ylabel("Score")
    ax2.set_title("Robustness Components")
    ax2.set_xticks(x)
    ax2.set_xticklabels(segment_ids, rotation=45, ha="right")
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_robustness_heatmap(
    robustness_scores: dict[str, RobustnessScore],
    *,
    title: str = "Robustness Score Heatmap",
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot robustness scores as a heatmap.

    Args:
        robustness_scores: Mapping of segment_id to RobustnessScore
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    segment_ids = list(robustness_scores.keys())
    metrics = ["Overall", "Feature Stability", "Time Consistency"]

    data = np.array([
        [r.overall_robustness, r.feature_stability, r.time_window_consistency]
        for r in robustness_scores.values()
    ])

    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        xticklabels=metrics,
        yticklabels=segment_ids,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()
    return fig


# =============================================================================
# ACTIONABILITY PLOTS
# =============================================================================


def plot_actionability_dimensions(
    evaluations: dict[str, ActionabilityEvaluation],
    *,
    title: str = "Actionability Dimensions Distribution",
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot distribution of actionability dimensions.

    Args:
        evaluations: Mapping of segment_id to ActionabilityEvaluation
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Count by dimension
    dimension_counts: dict[str, int] = {}
    for dim in ActionabilityDimension:
        dimension_counts[dim.value] = 0

    for evaluation in evaluations.values():
        for dim in evaluation.actionability_dimensions:
            dimension_counts[dim.value] += 1

    dims = list(dimension_counts.keys())
    counts = list(dimension_counts.values())

    ax1.bar(dims, counts, color=sns.color_palette("Set2", len(dims)))
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("Number of Segments")
    ax1.set_title("Segments by Actionability Dimension")

    # Right: Actionable vs Non-actionable
    actionable = sum(1 for e in evaluations.values() if e.is_actionable)
    not_actionable = len(evaluations) - actionable

    ax2.pie(
        [actionable, not_actionable],
        labels=["Actionable", "Not Actionable"],
        colors=["green", "red"],
        autopct="%1.0f%%",
        startangle=90,
    )
    ax2.set_title("Actionability Status")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_actionability_by_segment(
    evaluations: dict[str, ActionabilityEvaluation],
    *,
    title: str = "Actionability Dimensions by Segment",
    figsize: tuple[int, int] = (12, 8),
) -> Figure:
    """
    Plot actionability dimensions for each segment as a heatmap.

    Args:
        evaluations: Mapping of segment_id to ActionabilityEvaluation
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    segment_ids = list(evaluations.keys())
    dimensions = [d.value for d in ActionabilityDimension]

    # Create binary matrix
    data = []
    for evaluation in evaluations.values():
        row = [
            1 if ActionabilityDimension(dim) in evaluation.actionability_dimensions else 0
            for dim in dimensions
        ]
        data.append(row)

    sns.heatmap(
        data,
        annot=True,
        fmt="d",
        xticklabels=dimensions,
        yticklabels=segment_ids,
        cmap="YlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Has Dimension"},
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()
    return fig


# =============================================================================
# VIABILITY PLOTS
# =============================================================================


def plot_viability_scores(
    viabilities: dict[str, SegmentViability],
    *,
    title: str = "Segment Viability Assessment",
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """
    Plot viability scores for segments.

    Args:
        viabilities: Mapping of segment_id to SegmentViability
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    segment_ids = list(viabilities.keys())

    # Left: ROI comparison
    expected_roi = [v.expected_roi for v in viabilities.values()]

    colors = ["green" if roi > 1.0 else "orange" if roi > 0.5 else "red" for roi in expected_roi]
    ax1.barh(segment_ids, expected_roi, color=colors)
    ax1.axvline(x=1.0, color="green", linestyle="--", alpha=0.5, label="Break-even")
    ax1.set_xlabel("Expected ROI")
    ax1.set_title("Expected ROI by Segment")
    ax1.legend()

    # Right: Viability components radar (simplified as grouped bar)
    metrics = ["Marketing", "Sales", "Personalization", "Timing"]

    x = np.arange(len(segment_ids))
    width = 0.2

    marketing = [v.marketing_targetability for v in viabilities.values()]
    sales = [v.sales_prioritization for v in viabilities.values()]
    personalization = [v.personalization_opportunity for v in viabilities.values()]
    timing = [v.timing_optimization for v in viabilities.values()]

    ax2.bar(x - 1.5 * width, marketing, width, label="Marketing")
    ax2.bar(x - 0.5 * width, sales, width, label="Sales")
    ax2.bar(x + 0.5 * width, personalization, width, label="Personalization")
    ax2.bar(x + 1.5 * width, timing, width, label="Timing")

    ax2.set_xlabel("Segment")
    ax2.set_ylabel("Score")
    ax2.set_title("Viability Components")
    ax2.set_xticks(x)
    ax2.set_xticklabels(segment_ids, rotation=45, ha="right")
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# COMBINED DASHBOARD PLOTS
# =============================================================================


def plot_segment_dashboard(
    segments: list[Segment],
    *,
    robustness_scores: dict[str, RobustnessScore] | None = None,
    evaluations: dict[str, ActionabilityEvaluation] | None = None,
    title: str = "Segment Analysis Dashboard",
    figsize: tuple[int, int] = (16, 12),
) -> Figure:
    """
    Create a comprehensive dashboard view of segment analysis.

    Args:
        segments: List of segments
        robustness_scores: Optional robustness scores
        evaluations: Optional actionability evaluations
        title: Dashboard title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Top left: Segment sizes
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = [s.size for s in segments]
    names = [s.name[:20] for s in segments]  # Truncate names
    ax1.barh(names, sizes, color=sns.color_palette("viridis", len(segments)))
    ax1.set_xlabel("Customers")
    ax1.set_title("Segment Sizes")

    # Top middle: CLV distribution
    ax2 = fig.add_subplot(gs[0, 1])
    clv = [float(s.total_clv) for s in segments]
    ax2.bar(range(len(segments)), clv, color=sns.color_palette("viridis", len(segments)))
    ax2.set_xticks(range(len(segments)))
    ax2.set_xticklabels([s.name[:10] for s in segments], rotation=45, ha="right")
    ax2.set_ylabel("Total CLV ($)")
    ax2.set_title("Total CLV by Segment")

    # Top right: Robustness (if available)
    ax3 = fig.add_subplot(gs[0, 2])
    if robustness_scores:
        segment_ids = list(robustness_scores.keys())[:len(segments)]
        overall_scores = [robustness_scores[sid].overall_robustness for sid in segment_ids if sid in robustness_scores]
        colors = [
            "green" if robustness_scores[sid].robustness_tier == RobustnessTier.HIGH
            else "orange" if robustness_scores[sid].robustness_tier == RobustnessTier.MEDIUM
            else "red"
            for sid in segment_ids if sid in robustness_scores
        ]
        ax3.barh(segment_ids, overall_scores, color=colors)
        ax3.axvline(x=0.7, color="green", linestyle="--", alpha=0.5)
        ax3.axvline(x=0.4, color="red", linestyle="--", alpha=0.5)
        ax3.set_xlim(0, 1)
    ax3.set_xlabel("Robustness Score")
    ax3.set_title("Segment Robustness")

    # Bottom left: Actionability pie
    ax4 = fig.add_subplot(gs[1, 0])
    if evaluations:
        actionable = sum(1 for e in evaluations.values() if e.is_actionable)
        not_actionable = len(evaluations) - actionable
        ax4.pie(
            [actionable, not_actionable],
            labels=["Actionable", "Not Actionable"],
            colors=["green", "red"],
            autopct="%1.0f%%",
        )
    ax4.set_title("Actionability Distribution")

    # Bottom middle: Dimension distribution
    ax5 = fig.add_subplot(gs[1, 1])
    if evaluations:
        dimension_counts: dict[str, int] = {d.value: 0 for d in ActionabilityDimension}
        for e in evaluations.values():
            for dim in e.actionability_dimensions:
                dimension_counts[dim.value] += 1
        dims = list(dimension_counts.keys())
        counts = list(dimension_counts.values())
        ax5.bar(dims, counts, color=sns.color_palette("Set2", len(dims)))
    ax5.set_ylabel("Count")
    ax5.set_title("Actionability Dimensions")

    # Bottom right: Summary stats
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    total_customers = sum(s.size for s in segments)
    total_clv = sum(float(s.total_clv) for s in segments)
    actionable_count = len([e for e in (evaluations or {}).values() if e.is_actionable])

    summary_text = f"""
    Summary Statistics
    ─────────────────────
    Total Segments: {len(segments)}
    Total Customers: {total_customers:,}
    Total CLV: ${total_clv:,.2f}
    Actionable: {actionable_count} ({actionable_count/len(segments)*100:.0f}%)
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=12, family="monospace", va="center")

    fig.suptitle(title, fontsize=16, y=1.02)
    return fig


# =============================================================================
# REPORT VISUALIZATION
# =============================================================================


def plot_report_summary(
    report: SegmentationReport,
    *,
    figsize: tuple[int, int] = (14, 10),
) -> Figure:
    """
    Create visualization from SegmentationReport.

    Args:
        report: SegmentationReport to visualize
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top left: Segment sizes
    ax1 = fig.add_subplot(gs[0, 0])
    names = [sr.name[:15] for sr in report.segments]
    sizes = [sr.summary.size for sr in report.segments]
    ax1.barh(names, sizes, color=sns.color_palette("viridis", len(report.segments)))
    ax1.set_xlabel("Customers")
    ax1.set_title("Segment Sizes")

    # Top right: Robustness distribution
    ax2 = fig.add_subplot(gs[0, 1])
    robustness_dist = report.summary_stats.get("robustness_distribution", {})
    if robustness_dist:
        tiers = list(robustness_dist.keys())
        counts = list(robustness_dist.values())
        colors = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}
        ax2.bar(tiers, counts, color=[colors.get(t, "gray") for t in tiers])
    ax2.set_ylabel("Number of Segments")
    ax2.set_title("Robustness Distribution")

    # Bottom left: Actionability
    ax3 = fig.add_subplot(gs[1, 0])
    actionable = report.actionable_segments
    not_actionable = report.total_segments - actionable
    ax3.pie(
        [actionable, not_actionable],
        labels=["Actionable", "Not Actionable"],
        colors=["green", "red"],
        autopct="%1.0f%%",
    )
    ax3.set_title("Actionability Status")

    # Bottom right: Dimension distribution
    ax4 = fig.add_subplot(gs[1, 1])
    dim_dist = report.summary_stats.get("dimension_distribution", {})
    if dim_dist:
        dims = list(dim_dist.keys())
        counts = list(dim_dist.values())
        ax4.bar(dims, counts, color=sns.color_palette("Set2", len(dims)))
    ax4.set_ylabel("Count")
    ax4.set_title("Actionability Dimensions")

    fig.suptitle(report.title, fontsize=14, y=1.02)
    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def save_figure(
    fig: Figure,
    filepath: str,
    *,
    dpi: int = 150,
    bbox_inches: str = "tight",
) -> None:
    """
    Save figure to file.

    Args:
        fig: Figure to save
        filepath: Path to save to
        dpi: Resolution
        bbox_inches: Bounding box option
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)


def close_figure(fig: Figure) -> None:
    """Close a figure to free memory."""
    plt.close(fig)


def show_figure(fig: Figure) -> None:
    """Display figure (for notebooks)."""
    plt.show()
