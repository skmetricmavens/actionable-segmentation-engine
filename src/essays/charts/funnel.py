"""
Funnel chart specification for conversion visualization.

Used in Essay 3: The Illusion of New Customers
Shows the conversion funnel from new customer to retained.
"""

from typing import Any

from src.essays.base import ChartSpec
from src.essays.data_queries import NewCustomerQualityData


def create_funnel_spec(
    chart_id: str,
    data: NewCustomerQualityData,
    *,
    width: int = 600,
    height: int = 400,
) -> ChartSpec:
    """Create a conversion funnel specification.

    Args:
        chart_id: Unique identifier for the chart
        data: New customer quality data
        width: Chart width in pixels
        height: Chart height in pixels

    Returns:
        ChartSpec for D3 funnel rendering
    """
    # Build funnel stages with baseline context
    # Baseline is total_new_customers for percentage calculations
    baseline = data.total_new_customers

    stages = [
        {
            "id": "new",
            "label": "New Customers",
            "value": data.total_new_customers,
            "percentage": 100.0,
            "baseline": baseline,
            "display_text": f"100% ({data.total_new_customers:,} of {baseline:,})",
            "color": "#4299e1",
        },
        {
            "id": "second",
            "label": "Made 2nd Purchase",
            "value": data.customers_with_2nd_purchase,
            "percentage": data.conversion_to_2nd * 100,
            "baseline": baseline,
            "display_text": f"{data.conversion_to_2nd * 100:.0f}% ({data.customers_with_2nd_purchase:,} of {baseline:,})",
            "color": "#48bb78",
        },
        {
            "id": "third",
            "label": "Made 3rd Purchase",
            "value": data.customers_with_3rd_purchase,
            "percentage": data.conversion_to_3rd * 100,
            "baseline": baseline,
            "display_text": f"{data.conversion_to_3rd * 100:.0f}% ({data.customers_with_3rd_purchase:,} of {baseline:,})",
            "color": "#38a169",
        },
        {
            "id": "retained",
            "label": "Retained (3+ orders)",
            "value": data.retained_customers,
            "percentage": data.conversion_to_retained * 100,
            "baseline": baseline,
            "display_text": f"{data.conversion_to_retained * 100:.0f}% ({data.retained_customers:,} of {baseline:,})",
            "color": "#276749",
        },
    ]

    # Calculate drop-off between stages
    for i in range(1, len(stages)):
        prev = stages[i - 1]
        curr = stages[i]
        drop_off = prev["value"] - curr["value"]
        drop_off_pct = (drop_off / prev["value"] * 100) if prev["value"] > 0 else 0
        curr["drop_off"] = drop_off
        curr["drop_off_pct"] = drop_off_pct

    # Find the biggest leak
    max_drop_idx = 1
    max_drop = 0
    for i in range(1, len(stages)):
        if stages[i].get("drop_off", 0) > max_drop:
            max_drop = stages[i]["drop_off"]
            max_drop_idx = i

    # Scrolly triggers
    scrolly_triggers = [
        {
            "step": 1,
            "action": "show_stage",
            "params": {"stage_id": "new", "description": "All new customers start here"},
        },
        {
            "step": 2,
            "action": "show_stage",
            "params": {
                "stage_id": "second",
                "description": f"{data.conversion_to_2nd * 100:.0f}% make a second purchase",
            },
        },
        {
            "step": 3,
            "action": "show_stage",
            "params": {
                "stage_id": "third",
                "description": f"Only {data.conversion_to_3rd * 100:.0f}% reach their third order",
            },
        },
        {
            "step": 4,
            "action": "highlight_dropoff",
            "params": {
                "between": [stages[max_drop_idx - 1]["id"], stages[max_drop_idx]["id"]],
                "description": "Biggest leak in the funnel",
            },
        },
    ]

    return ChartSpec(
        chart_id=chart_id,
        chart_type="funnel",
        data={"stages": stages},
        config={
            "width": width,
            "height": height,
            "orientation": "vertical",
            "showLabels": True,
            "showPercentages": True,
            "showDropoffs": True,
            "dropoffColor": "#fc8181",
            "animate": True,
            "animationDuration": 800,
        },
        scrolly_triggers=scrolly_triggers,
        annotations=[
            {
                "type": "callout",
                "stage_id": stages[max_drop_idx]["id"],
                "text": f"Biggest drop: {stages[max_drop_idx].get('drop_off_pct', 0):.0f}%",
                "position": "right",
            },
        ],
    )


def create_quality_distribution_spec(
    chart_id: str,
    data: NewCustomerQualityData,
    *,
    width: int = 500,
    height: int = 300,
) -> ChartSpec:
    """Create a donut chart for quality score distribution.

    Args:
        chart_id: Unique identifier
        data: New customer quality data
        width: Chart width
        height: Chart height

    Returns:
        ChartSpec for donut chart
    """
    # Quality segments
    segments = [
        {
            "label": "High Potential",
            "value": data.quality_score_distribution.get("high", 0),
            "color": "#68d391",
        },
        {
            "label": "Medium Potential",
            "value": data.quality_score_distribution.get("medium", 0),
            "color": "#f6ad55",
        },
        {
            "label": "Low Potential",
            "value": data.quality_score_distribution.get("low", 0),
            "color": "#fc8181",
        },
    ]

    total = sum(s["value"] for s in segments)
    for segment in segments:
        segment["percentage"] = (segment["value"] / total * 100) if total > 0 else 0

    return ChartSpec(
        chart_id=chart_id,
        chart_type="donut",
        data={"segments": segments, "total": total},
        config={
            "width": width,
            "height": height,
            "innerRadius": 60,
            "outerRadius": 120,
            "showLabels": True,
            "showPercentages": True,
            "centerText": f"{total:,}",
            "centerSubtext": "New Customers",
        },
        scrolly_triggers=[
            {
                "step": 1,
                "action": "draw_segments",
                "params": {"duration": 1000},
            },
            {
                "step": 2,
                "action": "highlight_segment",
                "params": {"label": "High Potential"},
            },
        ],
    )


def create_ltv_histogram_spec(
    chart_id: str,
    data: NewCustomerQualityData,
    *,
    width: int = 600,
    height: int = 300,
) -> ChartSpec:
    """Create a histogram for LTV distribution.

    Args:
        chart_id: Unique identifier
        data: New customer quality data
        width: Chart width
        height: Chart height

    Returns:
        ChartSpec for histogram
    """
    # LTV buckets as bars
    bars = []
    for item in data.ltv_distribution:
        bars.append({
            "label": item["bucket"],
            "value": item["count"],
        })

    return ChartSpec(
        chart_id=chart_id,
        chart_type="bar",
        data={"bars": bars},
        config={
            "width": width,
            "height": height,
            "xLabel": "Predicted Lifetime Value",
            "yLabel": "Number of Customers",
            "color": "#4299e1",
            "showValues": True,
        },
        scrolly_triggers=[
            {
                "step": 1,
                "action": "draw_bars",
                "params": {"duration": 800, "stagger": 100},
            },
        ],
    )
