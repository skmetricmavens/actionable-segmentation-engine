"""
Analysis module for integrated segment analysis.

Provides unified analysis combining validation, actionability,
robustness, and whitespace opportunities.

Also includes trait discovery for identifying high-value product traits,
with explanations and user feedback learning loops.
"""

from src.analysis.integrated_analysis import (
    IntegratedAnalysisResult,
    IntegratedAnalyzer,
    SegmentWhitespace,
    UsableSegment,
    export_integrated_analysis,
    format_integrated_report,
)
from src.analysis.trait_discovery import (
    TraitDiscoveryResult,
    TraitMetadata,
    TraitValueAnalyzer,
    TraitValueScore,
    discover_traits,
    format_trait_report,
)
from src.analysis.trait_explainer import (
    TraitExplainer,
    TraitExplanation,
    explain_trait,
    format_explanation,
)
from src.analysis.trait_feedback import (
    FeedbackStore,
    LearnedPattern,
    TraitFeedback,
    load_feedback_store,
)

__all__ = [
    # Integrated analysis
    "IntegratedAnalysisResult",
    "IntegratedAnalyzer",
    "SegmentWhitespace",
    "UsableSegment",
    "export_integrated_analysis",
    "format_integrated_report",
    # Trait discovery
    "TraitDiscoveryResult",
    "TraitMetadata",
    "TraitValueAnalyzer",
    "TraitValueScore",
    "discover_traits",
    "format_trait_report",
    # Trait explanations
    "TraitExplainer",
    "TraitExplanation",
    "explain_trait",
    "format_explanation",
    # Trait feedback
    "FeedbackStore",
    "LearnedPattern",
    "TraitFeedback",
    "load_feedback_store",
]
