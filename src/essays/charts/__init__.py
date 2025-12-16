"""
D3 chart specifications for essay visualizations.

This module provides functions to create ChartSpec objects for each
visualization type. The specs contain the data and configuration that
the D3.js code needs to render the charts.

Each chart type has its own module with a create_*_spec function.
"""

from src.essays.charts.funnel import create_funnel_spec
from src.essays.charts.radar import create_radar_spec
from src.essays.charts.sankey import create_sankey_spec
from src.essays.charts.scatter import create_scatter_spec
from src.essays.charts.timeline import create_timeline_spec

__all__ = [
    "create_funnel_spec",
    "create_radar_spec",
    "create_sankey_spec",
    "create_scatter_spec",
    "create_timeline_spec",
]
