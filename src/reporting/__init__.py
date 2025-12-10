"""
Reporting module for the segmentation engine.

Contains report generation and visualization utilities.
"""

from src.reporting.segment_reporter import (
    ActionabilityReport,
    DecimalEncoder,
    ExplanationReport,
    RobustnessReport,
    SegmentationReport,
    SegmentReport,
    SegmentReporter,
    SegmentSummary,
    ViabilityReport,
    actionability_to_report,
    explanation_to_report,
    export_report_to_json,
    export_text_report,
    generate_segment_report,
    generate_segmentation_report,
    generate_text_summary,
    load_report_from_json,
    quick_report,
    report_to_dict,
    robustness_to_report,
    segment_to_summary,
    viability_to_report,
)
from src.reporting.visuals import (
    close_figure,
    plot_actionability_by_segment,
    plot_actionability_dimensions,
    plot_report_summary,
    plot_robustness_heatmap,
    plot_robustness_scores,
    plot_segment_dashboard,
    plot_segment_distribution,
    plot_segment_sizes_pie,
    plot_viability_scores,
    save_figure,
    set_style,
    show_figure,
)

__all__ = [
    # Report data classes
    "ActionabilityReport",
    "ExplanationReport",
    "RobustnessReport",
    "SegmentationReport",
    "SegmentReport",
    "SegmentSummary",
    "ViabilityReport",
    # Reporter class
    "SegmentReporter",
    # Conversion functions
    "actionability_to_report",
    "explanation_to_report",
    "robustness_to_report",
    "segment_to_summary",
    "viability_to_report",
    # Report generation
    "generate_segment_report",
    "generate_segmentation_report",
    "generate_text_summary",
    "quick_report",
    # Export functions
    "DecimalEncoder",
    "export_report_to_json",
    "export_text_report",
    "load_report_from_json",
    "report_to_dict",
    # Visualization functions
    "close_figure",
    "plot_actionability_by_segment",
    "plot_actionability_dimensions",
    "plot_report_summary",
    "plot_robustness_heatmap",
    "plot_robustness_scores",
    "plot_segment_dashboard",
    "plot_segment_distribution",
    "plot_segment_sizes_pie",
    "plot_viability_scores",
    "save_figure",
    "set_style",
    "show_figure",
]
