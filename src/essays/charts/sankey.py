"""
Sankey diagram specification for customer flow visualization.

Used in Essay 1: How Customers Actually Become Loyal
Shows the flow of customers between behavior types.
"""

from typing import Any

from src.essays.base import ChartSpec
from src.essays.data_queries import LoyaltyJourneyData


def create_sankey_spec(
    chart_id: str,
    data: LoyaltyJourneyData,
    *,
    width: int = 800,
    height: int = 500,
    node_width: int = 20,
    node_padding: int = 10,
) -> ChartSpec:
    """Create a Sankey diagram specification.

    Args:
        chart_id: Unique identifier for the chart
        data: Loyalty journey data with nodes and links
        width: Chart width in pixels
        height: Chart height in pixels
        node_width: Width of Sankey nodes
        node_padding: Padding between nodes

    Returns:
        ChartSpec for D3 Sankey rendering
    """
    # Build node list with indices and group assignments
    nodes = []
    node_index = {}

    for i, node in enumerate(data.sankey_nodes):
        node_id = node["id"]
        nodes.append({
            "id": node_id,
            "name": node["name"],
            "index": i,
            "group": node.get("group", 0),  # Column position for Sankey layout
        })
        node_index[node_id] = i

    # Build links with source/target indices
    links = []
    for link in data.sankey_links:
        source_id = link["source"]
        target_id = link["target"]

        if source_id in node_index and target_id in node_index:
            links.append({
                "source": node_index[source_id],
                "target": node_index[target_id],
                "value": link["value"],
                "source_id": source_id,
                "target_id": target_id,
            })

    # Scrolly triggers for animation
    scrolly_triggers = [
        {
            "step": 1,
            "action": "highlight_node",
            "params": {"node_id": "new", "description": "All customers start here"},
        },
        {
            "step": 2,
            "action": "highlight_flow",
            "params": {"source": "new", "target": "one_time", "description": "Many drop off immediately"},
        },
        {
            "step": 3,
            "action": "highlight_flow",
            "params": {"source": "new", "target": "regular", "description": "The path to loyalty"},
        },
        {
            "step": 4,
            "action": "show_all",
            "params": {"description": "The complete customer journey"},
        },
    ]

    # Color scheme for behavior types
    colors = {
        "new": "#4299e1",  # Blue
        "one_time": "#fc8181",  # Red
        "irregular": "#f6ad55",  # Orange
        "regular": "#68d391",  # Green
        "long_cycle": "#b794f4",  # Purple
    }

    return ChartSpec(
        chart_id=chart_id,
        chart_type="sankey",
        data={
            "nodes": nodes,
            "links": links,
        },
        config={
            "width": width,
            "height": height,
            "nodeWidth": node_width,
            "nodePadding": node_padding,
            "colors": colors,
            "linkOpacity": 0.5,
            "linkOpacityHover": 0.8,
        },
        scrolly_triggers=scrolly_triggers,
        annotations=[
            {
                "type": "label",
                "node_id": "regular",
                "text": "Champions",
                "position": "right",
            },
        ],
    )


def create_order_distribution_spec(
    chart_id: str,
    data: LoyaltyJourneyData,
    *,
    width: int = 600,
    height: int = 300,
) -> ChartSpec:
    """Create a bar chart spec for order count distribution.

    Args:
        chart_id: Unique identifier
        data: Loyalty journey data
        width: Chart width
        height: Chart height

    Returns:
        ChartSpec for D3 bar chart
    """
    # Convert distribution to bar data
    bars = []
    for order_count, customer_count in sorted(data.order_count_distribution.items()):
        label = f"{order_count}+" if order_count >= 10 else str(order_count)
        bars.append({
            "label": label,
            "value": customer_count,
            "order_count": order_count,
        })

    return ChartSpec(
        chart_id=chart_id,
        chart_type="bar",
        data={"bars": bars},
        config={
            "width": width,
            "height": height,
            "xLabel": "Number of Orders",
            "yLabel": "Customers",
            "color": "#4299e1",
            "highlightColor": "#2b6cb0",
        },
        scrolly_triggers=[
            {
                "step": 1,
                "action": "highlight_bar",
                "params": {"order_count": 1, "description": "One-time buyers"},
            },
            {
                "step": 2,
                "action": "highlight_bar",
                "params": {"order_count": data.loyalty_threshold_order, "description": "Loyalty threshold"},
            },
        ],
    )


def create_retention_line_spec(
    chart_id: str,
    data: LoyaltyJourneyData,
    *,
    width: int = 600,
    height: int = 300,
) -> ChartSpec:
    """Create a line chart spec for retention by order count.

    Args:
        chart_id: Unique identifier
        data: Loyalty journey data
        width: Chart width
        height: Chart height

    Returns:
        ChartSpec for D3 line chart
    """
    # Convert retention data to line points
    points = []
    for order_count, retention in sorted(data.retention_by_order_count.items()):
        points.append({
            "x": order_count,
            "y": retention * 100,  # Convert to percentage
            "label": f"Order {order_count}",
        })

    # Find threshold point for annotation
    threshold = data.loyalty_threshold_order
    threshold_retention = data.retention_by_order_count.get(threshold, 0) * 100

    return ChartSpec(
        chart_id=chart_id,
        chart_type="line",
        data={"points": points},
        config={
            "width": width,
            "height": height,
            "xLabel": "Order Number",
            "yLabel": "Retention Rate (%)",
            "color": "#68d391",
            "strokeWidth": 3,
            "showArea": True,
            "areaOpacity": 0.2,
        },
        scrolly_triggers=[
            {
                "step": 1,
                "action": "draw_line",
                "params": {"duration": 1000},
            },
            {
                "step": 2,
                "action": "add_annotation",
                "params": {
                    "x": threshold,
                    "y": threshold_retention,
                    "text": f"Threshold: Order {threshold}",
                },
            },
        ],
        annotations=[
            {
                "type": "vertical_line",
                "x": threshold,
                "label": f"Loyalty threshold",
                "style": "dashed",
            },
        ],
    )
