"""
Radar chart specification for multi-dimensional risk profiles.

Used in Essay 4: What Predicts Churn Before It Happens
Shows risk profile comparison between at-risk and all customers.
"""

from typing import Any

from src.essays.base import ChartSpec
from src.essays.data_queries import ChurnPredictionData


def create_radar_spec(
    chart_id: str,
    data: ChurnPredictionData,
    *,
    width: int = 500,
    height: int = 500,
) -> ChartSpec:
    """Create a radar chart specification for risk comparison.

    Args:
        chart_id: Unique identifier for the chart
        data: Churn prediction data with radar dimensions
        width: Chart width in pixels
        height: Chart height in pixels

    Returns:
        ChartSpec for D3 radar chart rendering
    """
    # Build radar axes from dimensions
    axes = []
    all_customer_values = []
    at_risk_values = []

    for dim in data.radar_dimensions:
        axes.append({
            "name": dim["dimension"],
            "max": 1.0,  # All values normalized to 0-1
        })
        all_customer_values.append(dim.get("all_customers", 0))
        at_risk_values.append(dim.get("at_risk", 0))

    # Two series: all customers vs at-risk
    series = [
        {
            "name": "All Customers",
            "values": all_customer_values,
            "color": "#4299e1",
            "fillOpacity": 0.2,
        },
        {
            "name": "At-Risk Customers",
            "values": at_risk_values,
            "color": "#fc8181",
            "fillOpacity": 0.3,
        },
    ]

    # Scrolly triggers
    scrolly_triggers = [
        {
            "step": 1,
            "action": "show_axes",
            "params": {"description": "Risk dimensions"},
        },
        {
            "step": 2,
            "action": "draw_series",
            "params": {
                "series_name": "All Customers",
                "description": "Average customer profile",
            },
        },
        {
            "step": 3,
            "action": "draw_series",
            "params": {
                "series_name": "At-Risk Customers",
                "description": "At-risk customers show elevated scores",
            },
        },
        {
            "step": 4,
            "action": "highlight_gap",
            "params": {"description": "The gap shows where at-risk customers differ most"},
        },
    ]

    # Find dimension with biggest gap
    max_gap_dim = None
    max_gap = 0
    for i, dim in enumerate(data.radar_dimensions):
        gap = abs(dim.get("at_risk", 0) - dim.get("all_customers", 0))
        if gap > max_gap:
            max_gap = gap
            max_gap_dim = dim["dimension"]

    return ChartSpec(
        chart_id=chart_id,
        chart_type="radar",
        data={
            "axes": axes,
            "series": series,
        },
        config={
            "width": width,
            "height": height,
            "levels": 5,  # Number of concentric circles
            "maxValue": 1.0,
            "labelFactor": 1.15,  # How far out labels are
            "wrapWidth": 80,  # Wrap axis labels at this width
            "opacityArea": 0.35,
            "dotRadius": 5,
            "strokeWidth": 2,
            "roundStrokes": True,
            "showLegend": True,
            "animate": True,
        },
        scrolly_triggers=scrolly_triggers,
        annotations=[
            {
                "type": "axis_highlight",
                "axis_name": max_gap_dim,
                "text": "Biggest difference",
            } if max_gap_dim else None,
        ],
    )


def create_at_risk_list_spec(
    chart_id: str,
    data: ChurnPredictionData,
    *,
    max_customers: int = 10,
) -> ChartSpec:
    """Create a ranked list specification for at-risk customers.

    Args:
        chart_id: Unique identifier
        data: Churn prediction data
        max_customers: Maximum customers to show

    Returns:
        ChartSpec for ranked list visualization
    """
    # Get top at-risk customers by churn risk
    top_at_risk = data.at_risk_customers[:max_customers]

    items = []
    for i, customer in enumerate(top_at_risk, 1):
        items.append({
            "rank": i,
            "id": customer["customer_id"],
            "risk_score": customer["churn_risk"],
            "clv": customer["clv"],
            "days_inactive": customer["days_since_purchase"],
            "purchases": customer["total_purchases"],
            # Risk level for color coding
            "risk_level": "high" if customer["churn_risk"] >= 0.7 else "medium",
        })

    return ChartSpec(
        chart_id=chart_id,
        chart_type="ranked_list",
        data={
            "items": items,
            "total_at_risk": data.total_at_risk,
        },
        config={
            "showRank": True,
            "showRiskBar": True,
            "showCLV": True,
            "columns": ["rank", "id", "risk_score", "clv", "days_inactive"],
            "columnLabels": {
                "rank": "#",
                "id": "Customer",
                "risk_score": "Risk",
                "clv": "CLV",
                "days_inactive": "Days Inactive",
            },
            "sortable": True,
            "defaultSort": "risk_score",
            "highlightThreshold": 0.7,
        },
        scrolly_triggers=[
            {
                "step": 1,
                "action": "reveal_rows",
                "params": {"duration": 100, "stagger": 50},
            },
            {
                "step": 2,
                "action": "highlight_high_risk",
                "params": {"threshold": 0.7},
            },
        ],
    )


def create_summary_metrics_spec(
    chart_id: str,
    data: ChurnPredictionData,
) -> ChartSpec:
    """Create summary metrics specification for churn overview.

    Args:
        chart_id: Unique identifier
        data: Churn prediction data

    Returns:
        ChartSpec for metrics display
    """
    metrics = [
        {
            "id": "at_risk_count",
            "label": "Customers at Risk",
            "value": data.total_at_risk,
            "format": ",d",
            "color": "#fc8181",
        },
        {
            "id": "at_risk_pct",
            "label": "% of Customer Base",
            "value": data.at_risk_percentage * 100,
            "format": ".1f",
            "suffix": "%",
            "color": "#f6ad55",
        },
        {
            "id": "revenue_at_risk",
            "label": "Revenue at Risk",
            "value": float(data.potential_revenue_loss),
            "format": ",.0f",
            "prefix": "$",
            "color": "#fc8181",
        },
        {
            "id": "warning_days",
            "label": "Avg Warning Window",
            "value": data.average_warning_days,
            "format": ".0f",
            "suffix": " days",
            "color": "#4299e1",
        },
    ]

    return ChartSpec(
        chart_id=chart_id,
        chart_type="metrics",
        data={"metrics": metrics},
        config={
            "layout": "horizontal",  # or "grid"
            "showIcons": True,
            "animate": True,
            "countUp": True,
            "countUpDuration": 2000,
        },
        scrolly_triggers=[
            {
                "step": 1,
                "action": "count_up",
                "params": {"duration": 2000},
            },
        ],
    )
