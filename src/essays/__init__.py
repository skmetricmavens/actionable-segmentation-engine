"""
Essays module for Pudding-style visual storytelling.

This module generates interactive, scrollytelling HTML essays from CRM data.
Essays transform segment metrics into narrative stories with D3.js visualizations.

Key components:
- base: Core dataclasses (Essay, EssaySection, ChartSpec, ChartNarrative)
- narratives: Auto-generate narrative text from data insights
- overrides: Load/apply manual narrative overrides from YAML
- charts: D3 chart specifications for each visualization type
- renderers: HTML rendering with Jinja2 templates
- sections: Essay section implementations (loyalty, champions, etc.)

Example usage:
    from src.essays import generate_essay, EssayConfig

    config = EssayConfig(
        essays=["loyalty", "champions"],
        audience="marketing",
        data_dir="data/samples",
        output_dir="output/essays",
    )
    essay = generate_essay(events, profiles, config)
    html = essay.render_html()
"""

from src.essays.base import (
    ChartNarrative,
    ChartSpec,
    Essay,
    EssayConfig,
    EssaySection,
    KeyInsight,
    ScrollyStep,
)

__all__ = [
    # Core dataclasses
    "ChartNarrative",
    "ChartSpec",
    "Essay",
    "EssayConfig",
    "EssaySection",
    "KeyInsight",
    "ScrollyStep",
]
