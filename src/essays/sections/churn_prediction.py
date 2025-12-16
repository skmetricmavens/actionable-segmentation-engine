"""
Essay 4: What Predicts Churn Before It Happens

This essay explores the leading indicators that appear before customers
stop buying, answering the question: What signals predict churn early
enough to intervene?

Key visualizations:
- Leading Indicator Timeline: Days-before-churn vs signal strength
- At-Risk Radar Chart: Customer risk profile across dimensions
- At-Risk Customer List: Prioritized intervention targets
"""

from src.essays.base import ChartNarrative, EssaySection, ScrollyStep
from src.essays.charts.radar import create_at_risk_list_spec, create_radar_spec
from src.essays.charts.timeline import create_timeline_spec
from src.essays.data_queries import ChurnPredictionData
from src.essays.narratives import (
    generate_churn_radar_narrative,
    generate_churn_timeline_narrative,
)


def create_churn_prediction_section(
    data: ChurnPredictionData,
    narratives: dict[str, ChartNarrative] | None = None,
) -> EssaySection:
    """Create the Churn Prediction essay section.

    Args:
        data: ChurnPredictionData with metrics and chart data
        narratives: Optional pre-generated narratives (with possible overrides)

    Returns:
        Complete EssaySection for the churn prediction essay
    """
    # Use provided narratives or generate fresh ones
    if narratives is None:
        narratives = {}

    timeline_narrative = narratives.get("churn_timeline") or generate_churn_timeline_narrative(data)
    radar_narrative = narratives.get("churn_radar") or generate_churn_radar_narrative(data)

    # Create chart specs
    timeline_chart = create_timeline_spec("churn_timeline", data)
    radar_chart = create_radar_spec("churn_radar", data)
    at_risk_list = create_at_risk_list_spec("at_risk_list", data)

    # Extract metrics
    warning_days = data.average_warning_days
    at_risk_count = data.total_at_risk
    at_risk_pct = data.at_risk_percentage * 100
    revenue_at_risk = float(data.potential_revenue_loss)

    # Get top predictors
    top_predictors = data.top_predictors[:3] if data.top_predictors else ["inactivity"]
    predictors_str = ", ".join(top_predictors)

    # Find the earliest warning indicator
    if data.leading_indicators:
        earliest = max(data.leading_indicators, key=lambda x: x.get("avg_days_before_churn", 0))
        earliest_days = earliest.get("avg_days_before_churn", 60)
        earliest_indicator = earliest.get("indicator", "behavioral changes")
    else:
        earliest_days = 60
        earliest_indicator = "behavioral changes"

    # Build scrollytelling steps
    scrolly_steps = [
        ScrollyStep(
            step_number=1,
            narrative_text=(
                f"Churn doesn't happen overnight. On average, warning signs appear "
                f"{warning_days:.0f} days before a customer makes their last purchase."
            ),
            chart_action="animateIn",
            action_params={},
        ),
        ScrollyStep(
            step_number=2,
            narrative_text=(
                f"The earliest signals appear around {earliest_days} days out. "
                f"'{earliest_indicator}' is often the first red flag."
            ),
            chart_action="highlightIndicator",
            action_params={"name": earliest_indicator},
        ),
        ScrollyStep(
            step_number=3,
            narrative_text=(
                f"The strongest predictors are: {predictors_str}. "
                f"These factors together explain most churn events."
            ),
            chart_action="showAll",
            action_params={},
        ),
        ScrollyStep(
            step_number=4,
            narrative_text=(
                f"Right now, {at_risk_count:,} customers ({at_risk_pct:.1f}%) show these patterns. "
                f"That's ${revenue_at_risk:,.0f} in potential revenue loss."
            ),
            chart_action="showAll",
            action_params={},
        ),
    ]

    # Build content layers
    executive_summary = (
        f"Warning signs appear {warning_days:.0f} days before churn. "
        f"{at_risk_count:,} customers are currently at risk (${revenue_at_risk:,.0f} potential loss)."
    )

    marketing_narrative = f"""
**The Predictive Window**

Churn doesn't happen suddenly. Our analysis shows that the average churning
customer displays warning signs {warning_days:.0f} days before their last purchase.
That's your intervention window.

**The Leading Indicators**

{_format_leading_indicators(data)}

The key insight: by the time RFM scores drop significantly, it's often too late.
These behavioral signals appear much earlier.

**Current At-Risk Customers**

Right now, {at_risk_count:,} customers ({at_risk_pct:.1f}% of your base) are showing
at-risk patterns. They represent ${revenue_at_risk:,.0f} in potential lost revenue.

**Intervention Strategy**

1. **Set up automated alerts**: Trigger outreach when key indicators appear
2. **Tiered response**: High-value at-risk customers get personal outreach;
   others get automated campaigns
3. **Test intervention timing**: Is day 30, 45, or 60 the optimal intervention point?
4. **Measure save rates**: Track what percentage of at-risk customers you retain

**Expected Impact**

Industry benchmarks suggest early intervention can save 20-40% of churning customers.
If you save even 20% of the ${revenue_at_risk:,.0f} at risk, that's ${revenue_at_risk * 0.2:,.0f}
in retained revenue.
"""

    technical_details = f"""
**Methodology**

Churn prediction based on analysis of customer behavior patterns. At-risk
customers identified using churn risk score threshold >= 0.5.

**Leading Indicators**
{_format_indicators_detailed(data)}

**At-Risk Customer Profile**

Total at risk: {at_risk_count:,} customers ({at_risk_pct:.1f}%)
Potential revenue loss: ${revenue_at_risk:,.2f}

**Risk Dimension Analysis**
{_format_radar_dimensions(data)}

**Top At-Risk Customers**
{_format_top_at_risk(data)}

**Model Performance Notes**
- Average warning days: {warning_days:.1f}
- Top predictors: {predictors_str}
- Risk threshold: 0.5 (churn risk score)
"""

    # Key metrics for this section
    key_metrics = {
        "warning_days": warning_days,
        "at_risk_count": at_risk_count,
        "at_risk_percentage": at_risk_pct,
        "revenue_at_risk": revenue_at_risk,
        "top_predictor": top_predictors[0] if top_predictors else "N/A",
        "earliest_warning_days": earliest_days,
    }

    return EssaySection(
        section_id="churn-prediction",
        title="What Predicts Churn Before It Happens",
        chart=timeline_chart,
        narrative=timeline_narrative,
        executive_summary=executive_summary,
        marketing_narrative=marketing_narrative,
        technical_details=technical_details,
        scrolly_steps=scrolly_steps,
        supporting_charts=[radar_chart, at_risk_list],
        key_metrics=key_metrics,
        appendix_data={
            "leading_indicators": data.leading_indicators,
            "radar_dimensions": data.radar_dimensions,
            "at_risk_customers": data.at_risk_customers[:20],  # Top 20
        },
    )


def _format_leading_indicators(data: ChurnPredictionData) -> str:
    """Format leading indicators for marketing narrative."""
    if not data.leading_indicators:
        return "No leading indicator data available."

    lines = []
    for i, ind in enumerate(data.leading_indicators[:5], 1):
        lines.append(
            f"{i}. **{ind['indicator']}**: Signals ~{ind['avg_days_before_churn']} days before churn. "
            f"{ind.get('description', '')}"
        )
    return "\n".join(lines)


def _format_indicators_detailed(data: ChurnPredictionData) -> str:
    """Format indicators with importance scores."""
    if not data.leading_indicators:
        return "No leading indicator data available."

    lines = []
    for ind in data.leading_indicators:
        importance_pct = ind.get("importance", 0) * 100
        lines.append(
            f"- {ind['indicator']}: {importance_pct:.0f}% importance, "
            f"~{ind['avg_days_before_churn']} days warning"
        )
    return "\n".join(lines)


def _format_radar_dimensions(data: ChurnPredictionData) -> str:
    """Format radar dimension comparison."""
    if not data.radar_dimensions:
        return "No radar dimension data available."

    lines = []
    for dim in data.radar_dimensions:
        all_score = dim.get("all_customers", 0) * 100
        at_risk_score = dim.get("at_risk", 0) * 100
        lines.append(
            f"- {dim['dimension']}: All customers {all_score:.0f}%, "
            f"At-risk {at_risk_score:.0f}%"
        )
    return "\n".join(lines)


def _format_top_at_risk(data: ChurnPredictionData) -> str:
    """Format top at-risk customers."""
    if not data.at_risk_customers:
        return "No at-risk customer data available."

    lines = ["(Top 5 by churn risk)"]
    for c in data.at_risk_customers[:5]:
        lines.append(
            f"- Customer {c['customer_id']}: "
            f"{c['churn_risk']:.0%} risk, "
            f"${c['clv']:,.0f} CLV, "
            f"{c['days_since_purchase']} days inactive"
        )
    return "\n".join(lines)
