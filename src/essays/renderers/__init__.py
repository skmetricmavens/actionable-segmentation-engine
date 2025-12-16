"""
HTML rendering for visual essays.

This module provides Jinja2-based HTML rendering for essays,
producing scrollytelling HTML with embedded D3 chart data.
"""

from src.essays.renderers.html import render_essay_html

__all__ = ["render_essay_html"]
