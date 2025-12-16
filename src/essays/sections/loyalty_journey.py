"""
Essay 1: How Customers Actually Become Loyal

This essay explores the customer journey from first purchase to loyal regular,
answering the question: When does a customer cross from "trying us out" to
"this is my store"?

Key visualizations:
- Sankey Flow: Customer journeys between behavior types
- Retention Line: Order # where retention probability jumps
- Order Distribution: Customer counts by number of orders
"""

from src.essays.base import ChartNarrative, EssaySection, ScrollyStep
from src.essays.charts.sankey import (
    create_order_distribution_spec,
    create_retention_line_spec,
    create_sankey_spec,
)
from src.essays.data_queries import LoyaltyJourneyData
from src.essays.narratives import (
    generate_loyalty_sankey_narrative,
    generate_loyalty_threshold_narrative,
)


def create_loyalty_journey_section(
    data: LoyaltyJourneyData,
    narratives: dict[str, ChartNarrative] | None = None,
) -> EssaySection:
    """Create the Loyalty Journey essay section.

    Args:
        data: LoyaltyJourneyData with metrics and chart data
        narratives: Optional pre-generated narratives (with possible overrides)

    Returns:
        Complete EssaySection for the loyalty journey essay
    """
    # Use provided narratives or generate fresh ones
    if narratives is None:
        narratives = {}

    sankey_narrative = narratives.get("loyalty_sankey") or generate_loyalty_sankey_narrative(data)
    threshold_narrative = narratives.get("loyalty_threshold") or generate_loyalty_threshold_narrative(data)

    # Create chart specs
    sankey_chart = create_sankey_spec("loyalty_sankey", data)
    retention_chart = create_retention_line_spec("loyalty_retention", data)
    distribution_chart = create_order_distribution_spec("loyalty_distribution", data)

    # Build scrollytelling steps
    one_time_pct = data.behavior_percentages.get("one_time", 0) * 100
    regular_pct = data.behavior_percentages.get("regular", 0) * 100
    threshold = data.loyalty_threshold_order
    threshold_retention = data.retention_by_order_count.get(threshold, 0) * 100

    scrolly_steps = [
        ScrollyStep(
            step_number=1,
            narrative_text=(
                f"Every customer starts their journey at the same place - their first purchase. "
                f"We analyzed {data.total_customers:,} customers to understand what happens next."
            ),
            chart_action="highlightNode",
            action_params={"nodeId": "new"},
        ),
        ScrollyStep(
            step_number=2,
            narrative_text=(
                f"The harsh reality: {one_time_pct:.0f}% of customers never make a second purchase. "
                f"They buy once and disappear."
            ),
            chart_action="highlightFlow",
            action_params={"source": "new", "target": "one_time"},
        ),
        ScrollyStep(
            step_number=3,
            narrative_text=(
                f"But here's the good news: once customers make it past their first few orders, "
                f"they tend to stick around. Only {regular_pct:.1f}% become loyal regulars, "
                f"but those who do are incredibly valuable."
            ),
            chart_action="highlightFlow",
            action_params={"source": "new", "target": "regular"},
        ),
        ScrollyStep(
            step_number=4,
            narrative_text=(
                f"The magic number is {threshold}. Customers who reach order #{threshold} "
                f"show a dramatic increase in retention - jumping to {threshold_retention:.0f}%."
            ),
            chart_action="showAll",
            action_params={},
        ),
    ]

    # Build content layers
    executive_summary = (
        f"Only {regular_pct:.1f}% of customers become loyal regulars. "
        f"Order #{threshold} is the critical loyalty threshold where retention jumps significantly."
    )

    marketing_narrative = f"""
**The Path to Customer Loyalty**

Our analysis of {data.total_customers:,} customers reveals a clear pattern: most customers
never make it past their first purchase. {one_time_pct:.0f}% buy once and never return.

However, the customers who do come back show dramatically different behavior.
Those who reach order #{threshold} have {threshold_retention:.0f}% retention rates -
they've crossed the threshold from "trying us out" to "this is my store."

**What This Means for Marketing**

The critical window is between orders 1 and {threshold}. This is where customers
decide if they're going to stay or leave. Focus your retention efforts here:

1. **First-purchase follow-up**: Send a compelling offer within 7 days of first purchase
2. **Order 2-{threshold} nurturing**: Create a specific campaign sequence for this group
3. **Loyalty recognition**: Celebrate when customers hit order #{threshold}

Champions average {data.average_orders_to_champion:.1f} orders. Once they cross the
loyalty threshold, they tend to keep buying.
"""

    technical_details = f"""
**Methodology**

Analysis based on {data.total_customers:,} customer profiles. Behavior types are
assigned based on purchase frequency, recency, and total order count.

**Behavior Distribution**
{_format_behavior_distribution(data)}

**Retention by Order Count**
{_format_retention_table(data)}

**Loyalty Threshold Detection**
The loyalty threshold (order #{threshold}) was identified by finding the order number
with the largest increase in retention probability compared to the previous order.
This represents the "inflection point" where customer behavior shifts from
transactional to loyal.
"""

    # Key metrics for this section
    key_metrics = {
        "total_customers": data.total_customers,
        "one_time_percentage": one_time_pct,
        "regular_percentage": regular_pct,
        "loyalty_threshold": threshold,
        "threshold_retention": threshold_retention,
        "average_orders_to_champion": data.average_orders_to_champion,
    }

    return EssaySection(
        section_id="loyalty-journey",
        title="How Customers Actually Become Loyal",
        chart=sankey_chart,
        narrative=sankey_narrative,
        executive_summary=executive_summary,
        marketing_narrative=marketing_narrative,
        technical_details=technical_details,
        scrolly_steps=scrolly_steps,
        supporting_charts=[retention_chart, distribution_chart],
        key_metrics=key_metrics,
        appendix_data={
            "behavior_counts": data.behavior_counts,
            "order_distribution": data.order_count_distribution,
            "retention_by_order": data.retention_by_order_count,
        },
    )


def _format_behavior_distribution(data: LoyaltyJourneyData) -> str:
    """Format behavior distribution as a text table."""
    lines = []
    for behavior, count in sorted(data.behavior_counts.items()):
        pct = data.behavior_percentages.get(behavior, 0) * 100
        lines.append(f"- {behavior.replace('_', ' ').title()}: {count:,} ({pct:.1f}%)")
    return "\n".join(lines)


def _format_retention_table(data: LoyaltyJourneyData) -> str:
    """Format retention by order count as a text table."""
    lines = []
    for orders, retention in sorted(data.retention_by_order_count.items())[:8]:
        lines.append(f"- Order {orders}: {retention * 100:.1f}% retention")
    return "\n".join(lines)
