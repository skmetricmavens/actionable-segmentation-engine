"""
Visualization catalog for Pudding.cool-style essay visualizations.

This module provides a queryable catalog of visualization patterns
that can be matched to data stories for optimal visualization selection.
"""

from src.essays.viz_catalog.matcher import (
    VizCatalog,
    VizPattern,
    match_visualization,
)

__all__ = [
    "VizCatalog",
    "VizPattern",
    "match_visualization",
]
