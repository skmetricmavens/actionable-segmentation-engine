"""
Scatter plot specification for quadrant visualizations.

Used in Essay 2: Not All Champions Are Safe
Shows revenue vs fragility for champion customers.
"""

from typing import Any

from src.essays.base import ChartSpec
from src.essays.data_queries import ChampionFragilityData


def create_scatter_spec(
    chart_id: str,
    data: ChampionFragilityData,
    *,
    width: int = 700,
    height: int = 500,
) -> ChartSpec:
    """Create a quadrant scatter plot specification.

    Args:
        chart_id: Unique identifier for the chart
        data: Champion fragility data with quadrant points
        width: Chart width in pixels
        height: Chart height in pixels

    Returns:
        ChartSpec for D3 scatter plot rendering
    """
    # Build point data
    points = []
    for point in data.quadrant_data:
        points.append({
            "x": point["x"],  # Revenue
            "y": point["y"],  # Fragility score
            "id": point["id"],
            "is_fragile": point["is_fragile"],
            "color": "#fc8181" if point["is_fragile"] else "#68d391",
        })

    # Calculate axis ranges
    revenues = [p["x"] for p in points] if points else [0]
    max_revenue = max(revenues) if revenues else 1000

    # Quadrant boundaries
    quadrant_config = {
        "x_threshold": max_revenue / 2,  # Mid-point for revenue
        "y_threshold": 0.5,  # Fragility threshold
        "quadrant_labels": {
            "top_left": "Low Value / High Risk",
            "top_right": "High Value / High Risk (Priority!)",
            "bottom_left": "Low Value / Stable",
            "bottom_right": "High Value / Stable (Ideal)",
        },
        "quadrant_colors": {
            "top_left": "rgba(252, 129, 129, 0.1)",
            "top_right": "rgba(252, 129, 129, 0.2)",  # Highlight danger zone
            "bottom_left": "rgba(160, 174, 192, 0.1)",
            "bottom_right": "rgba(104, 211, 145, 0.1)",
        },
    }

    # Scrolly triggers
    scrolly_triggers = [
        {
            "step": 1,
            "action": "show_points",
            "params": {"filter": "all", "description": "All champions"},
        },
        {
            "step": 2,
            "action": "show_quadrants",
            "params": {"description": "Revenue vs Fragility quadrants"},
        },
        {
            "step": 3,
            "action": "highlight_quadrant",
            "params": {
                "quadrant": "top_right",
                "description": "High-value, high-risk: these need immediate attention",
            },
        },
        {
            "step": 4,
            "action": "highlight_points",
            "params": {"filter": "fragile", "description": f"{data.fragile_champion_count} fragile champions"},
        },
    ]

    return ChartSpec(
        chart_id=chart_id,
        chart_type="scatter",
        data={"points": points},
        config={
            "width": width,
            "height": height,
            "xLabel": "Customer Revenue ($)",
            "yLabel": "Fragility Score",
            "xDomain": [0, max_revenue * 1.1],
            "yDomain": [0, 1],
            "pointRadius": 6,
            "pointOpacity": 0.7,
            "quadrants": quadrant_config,
            "tooltip": {
                "show": True,
                "fields": ["revenue", "fragility", "id"],
            },
        },
        scrolly_triggers=scrolly_triggers,
        annotations=[
            {
                "type": "horizontal_line",
                "y": 0.5,
                "label": "Risk threshold",
                "style": "dashed",
            },
        ],
    )


def create_thermometer_spec(
    chart_id: str,
    data: ChampionFragilityData,
    *,
    width: int = 300,
    height: int = 400,
) -> ChartSpec:
    """Create a risk thermometer visualization.

    Shows distribution of champions by risk level.

    Args:
        chart_id: Unique identifier
        data: Champion fragility data
        width: Chart width
        height: Chart height

    Returns:
        ChartSpec for thermometer visualization
    """
    # Risk buckets
    buckets = [
        {
            "level": "high",
            "label": "High Risk",
            "count": data.fragility_buckets.get("high", 0),
            "color": "#fc8181",
        },
        {
            "level": "medium",
            "label": "Medium Risk",
            "count": data.fragility_buckets.get("medium", 0),
            "color": "#f6ad55",
        },
        {
            "level": "low",
            "label": "Low Risk",
            "count": data.fragility_buckets.get("low", 0),
            "color": "#68d391",
        },
    ]

    total = sum(b["count"] for b in buckets)
    for bucket in buckets:
        bucket["percentage"] = (bucket["count"] / total * 100) if total > 0 else 0

    return ChartSpec(
        chart_id=chart_id,
        chart_type="thermometer",
        data={"buckets": buckets, "total": total},
        config={
            "width": width,
            "height": height,
            "orientation": "vertical",
            "showLabels": True,
            "showCounts": True,
            "animate": True,
        },
        scrolly_triggers=[
            {
                "step": 1,
                "action": "fill_thermometer",
                "params": {"duration": 1500},
            },
            {
                "step": 2,
                "action": "highlight_level",
                "params": {"level": "high", "pulse": True},
            },
        ],
    )
