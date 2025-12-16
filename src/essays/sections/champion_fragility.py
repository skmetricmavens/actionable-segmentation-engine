"""
Essay 2: Not All Champions Are Safe

This essay explores which high-value customers are at risk despite looking
great on paper, answering the question: Which Champions are fragile?

Key visualizations:
- Quadrant Scatter: Revenue (x) vs Fragility (y) with segment coloring
- Risk Thermometer: Champions ranked by decay probability
- Warning Signs list: Behavior changes that precede churn
"""

from src.essays.base import ChartNarrative, EssaySection, ScrollyStep
from src.essays.charts.scatter import create_scatter_spec, create_thermometer_spec
from src.essays.data_queries import ChampionFragilityData
from src.essays.narratives import (
    generate_champion_scatter_narrative,
    generate_champion_thermometer_narrative,
)


def create_champion_fragility_section(
    data: ChampionFragilityData,
    narratives: dict[str, ChartNarrative] | None = None,
) -> EssaySection:
    """Create the Champion Fragility essay section.

    Args:
        data: ChampionFragilityData with metrics and chart data
        narratives: Optional pre-generated narratives (with possible overrides)

    Returns:
        Complete EssaySection for the champion fragility essay
    """
    # Use provided narratives or generate fresh ones
    if narratives is None:
        narratives = {}

    scatter_narrative = narratives.get("champion_scatter") or generate_champion_scatter_narrative(data)
    thermometer_narrative = narratives.get("champion_thermometer") or generate_champion_thermometer_narrative(data)

    # Create chart specs
    scatter_chart = create_scatter_spec("champion_scatter", data)
    thermometer_chart = create_thermometer_spec("champion_thermometer", data)

    # Extract metrics
    fragile_pct = data.fragile_champion_percentage * 100
    fragile_count = data.fragile_champion_count
    total_champions = data.total_champions
    revenue_at_risk = float(data.fragile_revenue_at_risk)
    total_revenue = float(data.champion_revenue_total)
    high_risk = data.fragility_buckets.get("high", 0)
    medium_risk = data.fragility_buckets.get("medium", 0)

    # Build scrollytelling steps
    scrolly_steps = [
        ScrollyStep(
            step_number=1,
            narrative_text=(
                f"Your {total_champions:,} Champions represent your most valuable customers. "
                f"Together, they've generated ${total_revenue:,.0f} in revenue."
            ),
            chart_action="showAll",
            action_params={},
        ),
        ScrollyStep(
            step_number=2,
            narrative_text=(
                f"But not all Champions are equal. Look at the vertical axis - fragility score. "
                f"Higher scores mean higher risk of churning."
            ),
            chart_action="highlightAxis",
            action_params={"axis": "y"},
        ),
        ScrollyStep(
            step_number=3,
            narrative_text=(
                f"The danger zone: {fragile_count:,} Champions ({fragile_pct:.0f}%) are fragile. "
                f"They look great on paper but show warning signs."
            ),
            chart_action="highlightQuadrant",
            action_params={"quadrant": "top-right", "description": "High value, high risk"},
        ),
        ScrollyStep(
            step_number=4,
            narrative_text=(
                f"These fragile Champions represent ${revenue_at_risk:,.0f} in revenue at risk. "
                f"That's money that could walk out the door."
            ),
            chart_action="highlightFragile",
            action_params={},
        ),
    ]

    # Build content layers
    executive_summary = (
        f"{fragile_pct:.0f}% of Champions ({fragile_count:,} customers) are fragile. "
        f"${revenue_at_risk:,.0f} revenue at risk needs immediate attention."
    )

    marketing_narrative = f"""
**The Hidden Risk in Your Best Customers**

When we look at our top customers through traditional RFM metrics, everything looks
great. They've spent a lot, bought frequently, and their scores are high.

But dig deeper and a different picture emerges: {fragile_pct:.0f}% of these Champions
are showing warning signs of disengagement.

**What Makes a Champion Fragile?**

A fragile Champion typically shows one or more of these patterns:
- Haven't purchased in {data.days_threshold_for_risk}+ days (despite their history)
- Declining purchase frequency compared to their baseline
- Elevated churn risk score from behavioral signals

**The Revenue at Risk**

These {fragile_count:,} fragile Champions represent ${revenue_at_risk:,.0f} in potential
revenue loss. That's {revenue_at_risk / total_revenue * 100:.0f}% of your Champion revenue
that could disappear if we don't act.

**Recommended Actions**

1. **Immediate outreach**: Contact high-risk Champions this week with a personal touch
2. **Win-back offers**: Test incentives for Champions who haven't purchased in 30+ days
3. **Early warning system**: Set up alerts when Champion behavior changes
4. **Regular check-ins**: Schedule quarterly touchpoints with top Champions
"""

    technical_details = f"""
**Methodology**

Champions identified as customers with:
- Behavior type = Regular, OR
- Total revenue >= median and 3+ purchases

Fragility score calculation:
- 60% weight: Recency fragility (days since purchase / 90, capped at 1.0)
- 40% weight: Churn risk score from predictive model

Fragile threshold: fragility score > 0.5 OR days since purchase >= {data.days_threshold_for_risk}

**Risk Distribution**
- High risk (>0.6): {high_risk} Champions
- Medium risk (0.3-0.6): {medium_risk} Champions
- Low risk (<0.3): {data.fragility_buckets.get("low", 0)} Champions

**Warning Signs Analysis**
{_format_warning_signs(data)}

**Top Fragile Champions**
{_format_top_fragile(data)}
"""

    # Key metrics for this section
    key_metrics = {
        "total_champions": total_champions,
        "fragile_champions": fragile_count,
        "fragile_percentage": fragile_pct,
        "revenue_at_risk": revenue_at_risk,
        "total_champion_revenue": total_revenue,
        "high_risk_count": high_risk,
        "days_threshold": data.days_threshold_for_risk,
    }

    return EssaySection(
        section_id="champion-fragility",
        title="Not All Champions Are Safe",
        chart=scatter_chart,
        narrative=scatter_narrative,
        executive_summary=executive_summary,
        marketing_narrative=marketing_narrative,
        technical_details=technical_details,
        scrolly_steps=scrolly_steps,
        supporting_charts=[thermometer_chart],
        key_metrics=key_metrics,
        appendix_data={
            "fragility_buckets": data.fragility_buckets,
            "warning_signs": data.warning_signs,
            "top_fragile_champions": data.champions[:10] if data.champions else [],
        },
    )


def _format_warning_signs(data: ChampionFragilityData) -> str:
    """Format warning signs as text."""
    if not data.warning_signs:
        return "No warning signs data available."

    lines = []
    for sign in data.warning_signs:
        lines.append(
            f"- {sign['indicator']}: {sign['affected_count']} affected "
            f"({sign['risk_increase']:.1f}x risk increase)"
        )
    return "\n".join(lines)


def _format_top_fragile(data: ChampionFragilityData) -> str:
    """Format top fragile champions for technical appendix."""
    if not data.champions:
        return "No champion data available."

    lines = ["(Top 5 by fragility score)"]
    for c in data.champions[:5]:
        lines.append(
            f"- Customer {c['customer_id']}: "
            f"${c['revenue']:,.0f} revenue, "
            f"{c['fragility_score']:.2f} fragility, "
            f"{c['days_since_purchase']} days since purchase"
        )
    return "\n".join(lines)
