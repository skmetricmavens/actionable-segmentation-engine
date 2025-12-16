"""
Essay 3: The Illusion of New Customers

This essay explores whether acquisition efforts are bringing in customers
that actually stick, answering the question: Are we celebrating acquisition
that doesn't matter?

Key visualizations:
- Funnel Waterfall: New -> 2nd purchase -> 3rd purchase -> Retained
- Quality Score Distribution: New customers by predicted LTV
- LTV Distribution bars: Value breakdown of new customers
"""

from src.essays.base import ChartNarrative, EssaySection, ScrollyStep
from src.essays.charts.funnel import (
    create_funnel_spec,
    create_quality_distribution_spec,
)
from src.essays.data_queries import NewCustomerQualityData
from src.essays.narratives import (
    generate_funnel_narrative,
    generate_quality_distribution_narrative,
)


def create_new_customer_quality_section(
    data: NewCustomerQualityData,
    narratives: dict[str, ChartNarrative] | None = None,
) -> EssaySection:
    """Create the New Customer Quality essay section.

    Args:
        data: NewCustomerQualityData with metrics and chart data
        narratives: Optional pre-generated narratives (with possible overrides)

    Returns:
        Complete EssaySection for the new customer quality essay
    """
    # Use provided narratives or generate fresh ones
    if narratives is None:
        narratives = {}

    funnel_narrative = narratives.get("new_customer_funnel") or generate_funnel_narrative(data)
    quality_narrative = narratives.get("quality_distribution") or generate_quality_distribution_narrative(data)

    # Create chart specs
    funnel_chart = create_funnel_spec("new_customer_funnel", data)
    quality_chart = create_quality_distribution_spec("quality_distribution", data)

    # Extract metrics
    total = data.total_new_customers
    conv_2nd = data.conversion_to_2nd * 100
    conv_3rd = data.conversion_to_3rd * 100
    churn_rate = data.new_customer_churn_rate * 100
    avg_ltv = float(data.average_new_customer_ltv)

    high_quality = data.quality_score_distribution.get("high", 0)
    medium_quality = data.quality_score_distribution.get("medium", 0)
    low_quality = data.quality_score_distribution.get("low", 0)

    high_pct = (high_quality / total * 100) if total > 0 else 0
    low_pct = (low_quality / total * 100) if total > 0 else 0

    # Build scrollytelling steps
    scrolly_steps = [
        ScrollyStep(
            step_number=1,
            narrative_text=(
                f"We acquired {total:,} new customers. "
                f"That sounds like good news. But let's follow their journey."
            ),
            chart_action="highlightStage",
            action_params={"stageId": "new"},
        ),
        ScrollyStep(
            step_number=2,
            narrative_text=(
                f"Only {conv_2nd:.0f}% made a second purchase. "
                f"That's our first major drop-off point."
            ),
            chart_action="highlightDropoff",
            action_params={"fromId": "new", "toId": "second"},
        ),
        ScrollyStep(
            step_number=3,
            narrative_text=(
                f"And just {conv_3rd:.0f}% reached their third purchase - "
                f"the point where customers typically become valuable."
            ),
            chart_action="highlightStage",
            action_params={"stageId": "third"},
        ),
        ScrollyStep(
            step_number=4,
            narrative_text=(
                f"The bottom line: {churn_rate:.0f}% of new customers never return. "
                f"We need to rethink what 'successful acquisition' means."
            ),
            chart_action="showAll",
            action_params={},
        ),
    ]

    # Calculate drop-offs
    drop_1_to_2 = 100 - conv_2nd
    drop_2_to_3 = conv_2nd - conv_3rd if conv_2nd > 0 else 0

    # Build content layers
    executive_summary = (
        f"Only {conv_3rd:.0f}% of new customers reach their 3rd purchase. "
        f"{churn_rate:.0f}% never return after their first order."
    )

    marketing_narrative = f"""
**The New Customer Reality Check**

We celebrate acquisition numbers, but here's what those numbers really mean:
of {total:,} new customers, only {data.retained_customers:,} will become
valuable repeat buyers.

**The Funnel Leak**

- **After 1st purchase**: {drop_1_to_2:.0f}% never come back
- **After 2nd purchase**: {drop_2_to_3:.0f}% additional drop-off
- **Final retention**: Just {conv_3rd:.0f}% make it to order 3+

The biggest leak is between the first and second purchase. This is where
most customers decide we're not for them.

**Quality Over Quantity**

Not all new customers are created equal. Based on predicted lifetime value:
- **High potential**: {high_quality:,} customers ({high_pct:.0f}%)
- **Medium potential**: {medium_quality:,} customers
- **Low potential**: {low_quality:,} customers ({low_pct:.0f}%)

Average new customer LTV: ${avg_ltv:,.0f}

**Strategic Implications**

1. **First-purchase experience is critical**: What happens in the first 7 days
   largely determines if a customer returns

2. **Segment new customers by quality**: High-potential new customers deserve
   premium onboarding experiences

3. **Rethink CAC calculations**: If {churn_rate:.0f}% never return, your
   effective CAC is much higher than reported

4. **Consider acquisition channels**: Are some channels bringing higher-quality
   customers? (Requires channel attribution data)
"""

    technical_details = f"""
**Methodology**

Analysis based on {total:,} customer profiles across all acquisition cohorts.

**Conversion Funnel**
- New customers (1 purchase): {total:,} (100%)
- 2nd purchase: {data.customers_with_2nd_purchase:,} ({conv_2nd:.1f}%)
- 3rd purchase: {data.customers_with_3rd_purchase:,} ({conv_3rd:.1f}%)
- Retained (3+): {data.retained_customers:,} ({data.conversion_to_retained * 100:.1f}%)

**Quality Score Distribution**
Quality scores based on predicted Customer Lifetime Value (CLV):
- Low: Bottom 25% of CLV distribution
- Medium: 25th-75th percentile
- High: Top 25% of CLV distribution

**LTV Distribution**
{_format_ltv_distribution(data)}

**Key Metrics**
- Average new customer LTV: ${avg_ltv:,.2f}
- New customer churn rate: {churn_rate:.1f}%
- 1st-to-2nd purchase drop-off: {drop_1_to_2:.1f}%
- 2nd-to-3rd purchase drop-off: {drop_2_to_3:.1f}%
"""

    # Key metrics for this section
    key_metrics = {
        "total_new_customers": total,
        "conversion_to_2nd": conv_2nd,
        "conversion_to_3rd": conv_3rd,
        "churn_rate": churn_rate,
        "average_ltv": avg_ltv,
        "high_quality_pct": high_pct,
        "retained_customers": data.retained_customers,
    }

    return EssaySection(
        section_id="new-customer-quality",
        title="The Illusion of New Customers",
        chart=funnel_chart,
        narrative=funnel_narrative,
        executive_summary=executive_summary,
        marketing_narrative=marketing_narrative,
        technical_details=technical_details,
        scrolly_steps=scrolly_steps,
        supporting_charts=[quality_chart],
        key_metrics=key_metrics,
        appendix_data={
            "quality_distribution": data.quality_score_distribution,
            "ltv_distribution": data.ltv_distribution,
            "conversion_rates": {
                "to_2nd": conv_2nd,
                "to_3rd": conv_3rd,
                "to_retained": data.conversion_to_retained * 100,
            },
        },
    )


def _format_ltv_distribution(data: NewCustomerQualityData) -> str:
    """Format LTV distribution as text."""
    if not data.ltv_distribution:
        return "No LTV distribution data available."

    lines = []
    for bucket in data.ltv_distribution:
        lines.append(f"- {bucket['bucket']}: {bucket['count']:,} customers")
    return "\n".join(lines)
