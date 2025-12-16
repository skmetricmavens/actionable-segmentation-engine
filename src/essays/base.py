"""
Base dataclasses for Pudding-style visual essays.

This module defines the core data structures for essay generation:
- ChartNarrative: Auto-generated narrative with manual override support
- ChartSpec: D3 chart specification with data and config
- EssaySection: A single section with chart, narrative, and content layers
- Essay: Complete visual essay composed of sections
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass
class ChartNarrative:
    """Auto-generated narrative for a chart, with manual override support.

    The narrative explains the insight shown in a chart. It's auto-generated
    from data analysis but can be overridden via YAML config.
    """

    chart_id: str

    # Auto-generated from data analysis
    auto_headline: str = ""  # "Order #4 is your loyalty threshold"
    auto_insight: str = ""  # "Customers who reach 4 orders have 85% retention..."
    auto_callout: str = ""  # "This is 3x higher than customers with fewer orders"

    # Manual overrides (None = use auto-generated)
    override_headline: str | None = None
    override_insight: str | None = None
    override_callout: str | None = None

    @property
    def headline(self) -> str:
        """Get headline, preferring override if set."""
        return self.override_headline if self.override_headline is not None else self.auto_headline

    @property
    def insight(self) -> str:
        """Get insight text, preferring override if set."""
        return self.override_insight if self.override_insight is not None else self.auto_insight

    @property
    def callout(self) -> str:
        """Get callout text, preferring override if set."""
        return self.override_callout if self.override_callout is not None else self.auto_callout

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chart_id": self.chart_id,
            "headline": self.headline,
            "insight": self.insight,
            "callout": self.callout,
            "auto_headline": self.auto_headline,
            "auto_insight": self.auto_insight,
            "auto_callout": self.auto_callout,
            "has_overrides": any([
                self.override_headline is not None,
                self.override_insight is not None,
                self.override_callout is not None,
            ]),
        }


@dataclass
class ChartSpec:
    """Specification for a D3 chart.

    Contains the data and configuration needed to render a chart.
    The actual rendering is done by JavaScript; this just provides the spec.
    """

    chart_id: str
    chart_type: str  # "sankey", "scatter", "funnel", "timeline", "radar"

    # Data for the chart (will be JSON-serialized)
    data: dict[str, Any] = field(default_factory=dict)

    # Chart configuration
    config: dict[str, Any] = field(default_factory=dict)

    # Scrolly triggers: list of {step: int, action: str, params: dict}
    scrolly_triggers: list[dict[str, Any]] = field(default_factory=list)

    # Annotations to show on the chart
    annotations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "chart_id": self.chart_id,
            "chart_type": self.chart_type,
            "data": self.data,
            "config": self.config,
            "scrolly_triggers": self.scrolly_triggers,
            "annotations": self.annotations,
        }


@dataclass
class ScrollyStep:
    """A single step in a scrollytelling sequence."""

    step_number: int
    narrative_text: str  # Text shown alongside the chart
    chart_action: str | None = None  # Action to perform on chart (e.g., "highlight", "filter")
    action_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step_number,
            "text": self.narrative_text,
            "action": self.chart_action,
            "params": self.action_params,
        }


@dataclass
class EssaySection:
    """A single section of a visual essay.

    Each section contains:
    - A primary chart with narrative
    - Content at multiple audience layers (executive, marketing, technical)
    - Scrollytelling steps
    - Key metrics for the section
    """

    section_id: str
    title: str

    # Primary visualization
    chart: ChartSpec
    narrative: ChartNarrative

    # Content layers (progressive disclosure by audience)
    executive_summary: str = ""  # 1-2 sentences headline
    marketing_narrative: str = ""  # 2-3 paragraphs actionable story
    technical_details: str = ""  # Full methodology and stats

    # Scrollytelling sequence
    scrolly_steps: list[ScrollyStep] = field(default_factory=list)

    # Supporting visualizations (optional)
    supporting_charts: list[ChartSpec] = field(default_factory=list)

    # Key metrics for this section
    key_metrics: dict[str, Any] = field(default_factory=dict)

    # Raw data for technical appendix (optional)
    appendix_data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "chart": self.chart.to_dict(),
            "narrative": self.narrative.to_dict(),
            "executive_summary": self.executive_summary,
            "marketing_narrative": self.marketing_narrative,
            "technical_details": self.technical_details,
            "scrolly_steps": [s.to_dict() for s in self.scrolly_steps],
            "supporting_charts": [c.to_dict() for c in self.supporting_charts],
            "key_metrics": self.key_metrics,
        }


@dataclass
class KeyInsight:
    """A key insight to highlight in the executive summary banner."""

    text: str  # "23% of Champions are fragile"
    metric_value: str | float  # "23%" or 0.23
    metric_label: str  # "of Champions are fragile"
    category: str = "general"  # "loyalty", "risk", "acquisition", "churn"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "metric_value": self.metric_value,
            "metric_label": self.metric_label,
            "category": self.category,
        }


@dataclass
class Essay:
    """A complete visual essay composed of sections.

    This is the top-level structure that gets rendered to HTML.
    """

    essay_id: str
    title: str
    subtitle: str

    # Sections in order
    sections: list[EssaySection]

    # Key insights for executive summary banner
    key_insights: list[KeyInsight] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    data_period_start: date | None = None
    data_period_end: date | None = None
    customer_count: int = 0
    event_count: int = 0

    # Table of contents entries
    toc_entries: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        """Generate TOC from sections if not provided."""
        if not self.toc_entries and self.sections:
            self.toc_entries = [
                {"id": s.section_id, "title": s.title}
                for s in self.sections
            ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "essay_id": self.essay_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "sections": [s.to_dict() for s in self.sections],
            "key_insights": [i.to_dict() for i in self.key_insights],
            "generated_at": self.generated_at.isoformat(),
            "data_period_start": self.data_period_start.isoformat() if self.data_period_start else None,
            "data_period_end": self.data_period_end.isoformat() if self.data_period_end else None,
            "customer_count": self.customer_count,
            "event_count": self.event_count,
            "toc_entries": self.toc_entries,
        }


@dataclass
class EssayConfig:
    """Configuration for essay generation."""

    # Which essays to generate
    essays: list[str] = field(default_factory=lambda: ["all"])

    # Audience layer to emphasize
    audience: str = "all"  # "executive", "marketing", "technical", "all"

    # Data configuration
    data_dir: str = "data/samples"

    # Output configuration
    output_dir: str = "output/essays"

    # Override file path (optional)
    overrides_path: str | None = None

    # Whether to include technical appendix
    include_appendix: bool = True

    # Whether to embed assets or use CDN
    embed_assets: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "essays": self.essays,
            "audience": self.audience,
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "overrides_path": self.overrides_path,
            "include_appendix": self.include_appendix,
            "embed_assets": self.embed_assets,
        }
