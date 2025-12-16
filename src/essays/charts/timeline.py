"""
Timeline chart specification for leading indicators visualization.

Used in Essay 4: What Predicts Churn Before It Happens
Shows when warning signs appear before customer churn.
"""

from typing import Any

from src.essays.base import ChartSpec
from src.essays.data_queries import ChurnPredictionData


def create_timeline_spec(
    chart_id: str,
    data: ChurnPredictionData,
    *,
    width: int = 700,
    height: int = 400,
) -> ChartSpec:
    """Create a leading indicators timeline specification.

    Shows the timing and importance of churn warning signals.

    Args:
        chart_id: Unique identifier for the chart
        data: Churn prediction data
        width: Chart width in pixels
        height: Chart height in pixels

    Returns:
        ChartSpec for D3 timeline rendering
    """
    # Build indicator data points
    indicators = []
    for ind in data.leading_indicators:
        indicators.append({
            "name": ind["indicator"],
            "days_before": ind["avg_days_before_churn"],
            "importance": ind["importance"],
            "description": ind["description"],
            # Size based on importance, position based on timing
            "radius": 10 + ind["importance"] * 30,
            "color": _importance_to_color(ind["importance"]),
        })

    # Sort by days before churn (furthest first)
    indicators.sort(key=lambda x: -x["days_before"])

    # Scrolly triggers - reveal indicators one by one
    scrolly_triggers = [
        {
            "step": 1,
            "action": "show_timeline",
            "params": {"description": "Days before churn"},
        },
    ]

    for i, ind in enumerate(indicators[:5], start=2):
        scrolly_triggers.append({
            "step": i,
            "action": "show_indicator",
            "params": {
                "name": ind["name"],
                "description": f"{ind['name']}: appears ~{ind['days_before']} days before churn",
            },
        })

    scrolly_triggers.append({
        "step": len(indicators[:5]) + 2,
        "action": "show_all",
        "params": {"description": f"You have {data.average_warning_days:.0f} days on average to intervene"},
    })

    return ChartSpec(
        chart_id=chart_id,
        chart_type="timeline",
        data={
            "indicators": indicators,
            "average_warning_days": data.average_warning_days,
        },
        config={
            "width": width,
            "height": height,
            "xLabel": "Days Before Churn",
            "xDomain": [90, 0],  # Reverse: further out on left
            "showImportance": True,
            "importanceLabel": "Signal Strength",
            "animate": True,
            "showConnections": True,
        },
        scrolly_triggers=scrolly_triggers,
        annotations=[
            {
                "type": "vertical_line",
                "x": data.average_warning_days,
                "label": f"Average warning: {data.average_warning_days:.0f} days",
                "style": "dashed",
                "color": "#f6ad55",
            },
            {
                "type": "zone",
                "x_start": 14,
                "x_end": 0,
                "label": "Critical window",
                "color": "rgba(252, 129, 129, 0.2)",
            },
        ],
    )


def _importance_to_color(importance: float) -> str:
    """Convert importance score to color.

    Higher importance = more red/urgent.
    Lower importance = more blue/cool.
    """
    if importance >= 0.3:
        return "#fc8181"  # Red - high importance
    elif importance >= 0.15:
        return "#f6ad55"  # Orange - medium importance
    else:
        return "#4299e1"  # Blue - lower importance


def create_importance_bar_spec(
    chart_id: str,
    data: ChurnPredictionData,
    *,
    width: int = 500,
    height: int = 300,
) -> ChartSpec:
    """Create a horizontal bar chart for indicator importance.

    Args:
        chart_id: Unique identifier
        data: Churn prediction data
        width: Chart width
        height: Chart height

    Returns:
        ChartSpec for horizontal bar chart
    """
    bars = []
    for ind in data.leading_indicators:
        bars.append({
            "label": ind["indicator"],
            "value": ind["importance"] * 100,  # Convert to percentage
            "color": _importance_to_color(ind["importance"]),
        })

    # Sort by importance
    bars.sort(key=lambda x: -x["value"])

    return ChartSpec(
        chart_id=chart_id,
        chart_type="bar_horizontal",
        data={"bars": bars},
        config={
            "width": width,
            "height": height,
            "xLabel": "Predictive Importance (%)",
            "showValues": True,
            "valueFormat": ".0f",
            "animate": True,
        },
        scrolly_triggers=[
            {
                "step": 1,
                "action": "draw_bars",
                "params": {"duration": 800, "stagger": 150},
            },
            {
                "step": 2,
                "action": "highlight_top",
                "params": {"count": 3, "description": "Top 3 predictors"},
            },
        ],
    )
